---
description: "Genera un Roadmap Técnico en formato Markdown a partir del plan del proyecto actual"
mode: "agent"
---

Actúa como un Technical Project Manager. Toma el **plan del proyecto que acabas de crear** en esta misma conversación (el `implementation_plan.md` o artefacto de planificación que ya está en tu contexto) y transfórmalo en un **Roadmap Técnico** estructurado en Sprints.

## Instrucciones

1. **Conservar el plan original:** Copia el plan original tal cual a `PLAN_ORIGINAL.md` en la raíz del proyecto, para preservar la versión íntegra generada por la IA antes de la transformación.
2. **Descomponer en Sprints:** Analiza el plan (que ya tienes en contexto) y agrupa las tareas en iteraciones secuenciales (Sprints), cada uno con un objetivo claro y acotado.
3. **Desglose de Trabajo (WBS):** Descompón cada Sprint en tareas y subtareas accionables usando casillas de verificación (`- [ ]`).
4. **Criterios de Aceptación (Tests):** Asigna a cada tarea los tests de verificación exactos que deben ejecutarse para validarla. Deben ser comandos concretos y ejecutables siempre que sea posible.
5. **Definition of Done (DoD):** Incluye estas reglas innegociables:
   - *Nivel Tarea:* Una tarea solo se marca como completa si sus tests de verificación pasan al 100%.
   - *Nivel Sprint:* Un Sprint solo se archiva como completado tras verificar con éxito el 100% de sus tareas.
6. **Generar archivo:** Escribe el roadmap resultante en `ROADMAP.md` en la raíz del proyecto.

## Formato de salida requerido (aplica esta plantilla exacta):

```markdown
# 🗺️ Roadmap del Proyecto

> **Definition of Done (DoD):** Toda tarea requiere un 100% de éxito en sus tests para cerrarse. Un Sprint se archiva únicamente al verificar el 100% de sus tareas.

---

## Sprint [N]: [Objetivo del Sprint]

- [ ] **Tarea [N.1]: [Nombre de la Tarea]**
  - *Descripción:* [Breve contexto técnico]
  - *Subtareas:*
    - [ ] Subtarea 1: [Detalle]
    - [ ] Subtarea 2: [Detalle]
  - 🧪 *Tests de Verificación (Requisito 100% Pass):*
    - [ ] Test 1: [Condición a evaluar]
    - [ ] Test 2: [Condición a evaluar]

- [ ] **Tarea [N.2]: [Nombre de la Tarea]**
  - *Descripción:* [Breve contexto técnico]
  - *Subtareas:*
    - [ ] Subtarea 1: [Detalle]
    - [ ] Subtarea 2: [Detalle]
  - 🧪 *Tests de Verificación (Requisito 100% Pass):*
    - [ ] Test 1: [Condición a evaluar]
    - [ ] Test 2: [Condición a evaluar]
```
