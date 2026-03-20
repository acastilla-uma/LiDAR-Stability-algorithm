#!/usr/bin/env python3
"""
Download CNIG LiDAR 3rd coverage tiles from a list of file names.

Flow implemented from CNIG page logic:
  1) Search file in `archivosSerie` with keySearch -> resolve `sec`
  2) Call `initDescargaDir?secuencial=<sec>`
  3) POST `descargaDir` with `secDescDirLA=<secuencialDescDir>`

Usage example:
  python src/lidar_stability/lidar/download_cnig_lidar_tiles.py \
    --list output/cnig_missing_tiles_all_doback.txt \
    --dest LiDAR-Maps/cnig
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


ARCHIVOS_SERIE_PARAMS_BASE = {
    "numPagina": "1",
    "codAgr": "MOMDT",
    "codSerie": "LIDA3",
    "coordenadas": "",
    "series": "",
    "codComAutonoma": "",
    "codProvincia": "",
    "codIne": "",
    "codTipoArchivo": "",
    "codIdiomaInf": "",
    "todaEspania": "",
    "todoMundo": "",
    "idProductor": "",
    "rutaNombre": "",
    "numHoja": "",
    "numHoja25": "",
    "totalArchivos": "290431",
    "codSubSerie": "",
    "contieneArc": "",
    "referCatastral": "",
    "orderBy": "",
}


@dataclass
class DownloadResult:
    requested: str
    status: str
    detail: str
    saved_as: Optional[str] = None


def normalize_tile_name(name: str) -> str:
    value = name.strip()
    value = value.replace("_", "-")
    value = value.upper()
    if value.endswith(".LAZ"):
        return value
    if value.endswith(".LAS"):
        return value[:-4] + ".LAZ"
    return value + ".LAZ"


def expected_local_variants(requested_name: str) -> List[str]:
    raw = requested_name.strip()
    if not raw:
        return []

    variants = {raw}
    upper = raw.upper()
    if not upper.endswith(".LAZ"):
        if upper.endswith(".LAS"):
            upper = upper[:-4] + ".LAZ"
        else:
            upper += ".LAZ"

    variants.add(upper)
    variants.add(upper.lower())
    variants.add(upper.replace("-", "_"))
    variants.add(upper.replace("-", "_").lower())
    variants.add(upper.replace("_", "-"))
    variants.add(upper.replace("_", "-").lower())

    return sorted(v for v in variants if v)


def parse_content_disposition_filename(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None

    match = re.search(r'filename\*=UTF-8\'\'([^;]+)', header_value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"')

    match = re.search(r'filename=([^;]+)', header_value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"')

    return None


def read_tile_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"List file not found: {path}")

    items: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        items.append(value)
    return items


class CNIGLidarDownloader:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        retries: int,
        backoff_s: float,
        delay_s: float,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff_s = backoff_s
        self.delay_s = delay_s
        self.opener = build_opener(HTTPCookieProcessor())
        self._search_cache: Dict[str, Dict[str, str]] = {}

    def _request(self, path: str, params: Optional[Dict[str, str]] = None, data: Optional[Dict[str, str]] = None) -> bytes:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CNIG-LiDAR-Downloader/1.0",
            "Accept": "*/*",
        }

        if params:
            query = urlencode(params)
            connector = "&" if "?" in url else "?"
            url = f"{url}{connector}{query}"

        payload = None
        if data is not None:
            payload = urlencode(data).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                req = Request(url=url, data=payload, headers=headers)
                with self.opener.open(req, timeout=self.timeout) as resp:
                    body = resp.read()
                    req.headers["_resp_content_disposition"] = resp.headers.get("Content-Disposition", "")
                    req.headers["_resp_content_type"] = resp.headers.get("Content-Type", "")
                    req.headers["_status"] = str(getattr(resp, "status", ""))
                    return body, req.headers
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                wait_s = self.backoff_s * attempt
                logger.warning(f"Request failed ({path}) attempt {attempt}/{self.retries}: {exc}. Retrying in {wait_s:.1f}s")
                time.sleep(wait_s)

        raise RuntimeError(f"Request failed after {self.retries} attempts: {path}: {last_error}")

    def open_session(self) -> None:
        self._request("lidar-tercera-cobertura")

    def query_filename_to_sec_map(self, tile_name: str) -> Dict[str, str]:
        canonical = normalize_tile_name(tile_name)
        if canonical in self._search_cache:
            return self._search_cache[canonical]

        params = dict(ARCHIVOS_SERIE_PARAMS_BASE)
        params["keySearch"] = canonical.replace("_", "-")

        body, _headers = self._request("archivosSerie", params=params)
        html = body.decode("utf-8", errors="ignore")

        row_pattern = re.compile(r"<tr class=\"fontSize08em row100\">(.*?)</tr>", flags=re.DOTALL | re.IGNORECASE)
        name_pattern = re.compile(
            r'<div class=\"col-m-8 lineHeight30 displayInlineBlock txtLeftCenterTablas\">\s*([^<]+?)\s*</div>',
            flags=re.IGNORECASE,
        )
        sec_pattern = re.compile(r'id=\"linkDescDir_(\d+)\"', flags=re.IGNORECASE)

        resolved: Dict[str, str] = {}
        for row_match in row_pattern.finditer(html):
            row_html = row_match.group(1)
            name_match = name_pattern.search(row_html)
            sec_match = sec_pattern.search(row_html)
            if not name_match or not sec_match:
                continue
            found_name = normalize_tile_name(name_match.group(1))
            resolved[found_name] = sec_match.group(1)

        self._search_cache[canonical] = resolved
        return resolved

    def resolve_sec(self, requested_name: str) -> Optional[Tuple[str, str]]:
        target = normalize_tile_name(requested_name)
        resolved_map = self.query_filename_to_sec_map(target)
        if target in resolved_map:
            return target, resolved_map[target]

        if len(resolved_map) == 1:
            found_name, sec = next(iter(resolved_map.items()))
            return found_name, sec

        return None

    def init_download(self, sec: str) -> Dict[str, str]:
        body, _headers = self._request("initDescargaDir", params={"secuencial": sec})
        payload = json.loads(body.decode("utf-8", errors="ignore"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected initDescargaDir response for sec={sec}: {payload}")
        return payload

    def fetch_file(self, sec_descarga: str) -> Tuple[bytes, Dict[str, str]]:
        return self._request("descargaDir", data={"secDescDirLA": sec_descarga})


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Descarga teselas LAZ de CNIG (LiDAR 3Âª cobertura) a partir de una lista de nombres."
    )
    parser.add_argument(
        "--list",
        type=Path,
        default=Path("output/cnig_missing_tiles_all_doback.txt"),
        help="Fichero con nombres de teselas (.laz), una por lÃ­nea.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("LiDAR-Maps/cnig"),
        help="Directorio de destino para los ficheros descargados.",
    )
    parser.add_argument(
        "--base-url",
        default="https://centrodedescargas.cnig.es/CentroDescargas",
        help="URL base del Centro de Descargas CNIG.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout por peticiÃ³n (segundos).")
    parser.add_argument("--retries", type=int, default=3, help="Reintentos por peticiÃ³n.")
    parser.add_argument("--backoff", type=float, default=2.0, help="Backoff base entre reintentos (segundos).")
    parser.add_argument("--delay", type=float, default=0.2, help="Pausa entre ficheros (segundos).")
    parser.add_argument("--max-files", type=int, default=0, help="MÃ¡ximo de ficheros a procesar (0 = todos).")
    parser.add_argument("--dry-run", action="store_true", help="Resuelve secuenciales, pero no descarga.")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe ficheros ya existentes.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/cnig_download_report.csv"),
        help="CSV de reporte final (estado por tesela).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    args.dest.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    requested = read_tile_list(args.list)
    if args.max_files > 0:
        requested = requested[: args.max_files]

    downloader = CNIGLidarDownloader(
        base_url=args.base_url,
        timeout=args.timeout,
        retries=max(1, args.retries),
        backoff_s=max(0.0, args.backoff),
        delay_s=max(0.0, args.delay),
    )

    logger.info(f"Tiles in list: {len(requested)}")
    downloader.open_session()

    results: List[DownloadResult] = []
    for index, tile in enumerate(requested, start=1):
        tile = tile.strip()
        if not tile:
            continue

        logger.info(f"[{index}/{len(requested)}] {tile}")

        existing_path = None
        if not args.overwrite:
            for name_variant in expected_local_variants(tile):
                candidate = args.dest / name_variant
                if candidate.exists() and candidate.stat().st_size > 0:
                    existing_path = candidate
                    break
        if existing_path is not None:
            results.append(DownloadResult(tile, "skipped_exists", f"already exists", str(existing_path.name)))
            continue

        try:
            resolved = downloader.resolve_sec(tile)
        except Exception as exc:
            results.append(DownloadResult(tile, "error", f"search failed: {exc}"))
            continue

        if resolved is None:
            results.append(DownloadResult(tile, "not_found", "no matching sec found"))
            continue

        resolved_name, sec = resolved

        if args.dry_run:
            results.append(DownloadResult(tile, "resolved", f"sec={sec}", resolved_name))
            continue

        try:
            init_payload = downloader.init_download(sec)
            sec_desc = str(init_payload.get("secuencialDescDir", "")).strip()
            muestra_lic = str(init_payload.get("muestraLic", "")).strip().upper()
            if not sec_desc:
                results.append(DownloadResult(tile, "error", f"init missing secuencialDescDir (sec={sec})"))
                continue
            if muestra_lic == "SI":
                results.append(DownloadResult(tile, "needs_license", f"manual acceptance required (sec={sec_desc})"))
                continue

            content, headers = downloader.fetch_file(sec_desc)
            if not content:
                results.append(DownloadResult(tile, "error", "empty download response"))
                continue

            cd_header = headers.get("_resp_content_disposition", "")
            server_name = parse_content_disposition_filename(cd_header)
            output_name = server_name or tile
            if not output_name.lower().endswith(".laz"):
                output_name += ".laz"
            output_path = args.dest / output_name

            output_path.write_bytes(content)
            results.append(DownloadResult(tile, "downloaded", f"bytes={len(content)} sec={sec_desc}", output_name))

        except Exception as exc:
            results.append(DownloadResult(tile, "error", f"download failed: {exc}"))

        if downloader.delay_s > 0:
            time.sleep(downloader.delay_s)

    # Write CSV report
    lines = ["requested,status,detail,saved_as"]
    for item in results:
        req = item.requested.replace('"', "''")
        st = item.status.replace('"', "''")
        det = item.detail.replace('"', "''")
        sav = (item.saved_as or "").replace('"', "''")
        lines.append(f'"{req}","{st}","{det}","{sav}"')
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = len(results)
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    logger.info("--- SUMMARY ---")
    logger.info(f"Processed: {total}")
    for key in sorted(counts.keys()):
        logger.info(f"{key}: {counts[key]}")
    logger.info(f"Report: {args.report}")

    # Non-zero if any hard error
    return 1 if counts.get("error", 0) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

