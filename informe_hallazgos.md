# Informe de Hallazgos: Análisis de Desechos Textiles en Latinoamérica

## Taller 1 — Toma de Decisiones II (2026-01)
**Industria Textil — Consultora de Sostenibilidad**

---

## 1. Descripción del Dataset

El dataset contiene **500 observaciones** y **13 variables** sobre desechos textiles en ciudades latinoamericanas.

| Tipo | Variables |
|------|-----------|
| Dependiente | Toneladas (desechos textiles) |
| Continuas independientes (10) | Ventas, Produccion, Inventario, Algodón, Nailon, Poliéster, Energia, Productividad, Proteccion, Agua |
| Categóricas (2) | Ciudad (Bogota, Medellin, Mexico, Santiago), Mes (10 meses) |

- No se encontraron valores nulos en ninguna columna.
- Santiago concentra la mayor cantidad de observaciones (200), seguida de Bogotá (131), México (125) y Medellín (44).
- Los meses con más registros son Abril (113), Marzo (85) y Febrero (79). Octubre y Septiembre tienen muy pocos datos (3 y 4 respectivamente).

---

## 2. Análisis Descriptivo (Punto 1)

### Estadísticas principales de la variable dependiente

| Estadístico | Toneladas |
|-------------|-----------|
| Media | 393.46 |
| Desviación estándar | 126.12 |
| Mínimo | 2.17 |
| Q1 (25%) | 305.82 |
| Mediana | 391.98 |
| Q3 (75%) | 473.73 |
| Máximo | 809.57 |
| CV% | 32.05% |
| Asimetría | 0.076 |
| Curtosis | 0.105 |

La distribución de Toneladas es aproximadamente simétrica (asimetría ≈ 0) y mesocúrtica (curtosis ≈ 0), lo que sugiere una distribución cercana a la normal.

### Distribución de Toneladas (histograma, boxplot, QQ-plot)

![Distribución de Toneladas](img/grafica_1.png)

La media (393.5) y la mediana (392.0) son muy cercanas, confirmando la simetría. El QQ-plot muestra un buen ajuste a la distribución normal.

### Distribución de todas las variables numéricas

![Distribución de variables numéricas](img/grafica_2.png)

Todas las variables numéricas presentan distribuciones razonablemente simétricas. No se observan distribuciones severamente sesgadas.

### Boxplots de variables numéricas

![Boxplots](img/grafica_3.png)

Se identifican algunos valores atípicos leves en varias variables, pero ninguno extremo que requiera tratamiento especial.

### Análisis por variables categóricas

![Toneladas por Ciudad y Mes](img/grafica_4.png)

**Promedio de Toneladas por Ciudad:**

| Ciudad | Media | Desv. Est. | n |
|--------|-------|------------|---|
| Santiago | 414.81 | 126.88 | 200 |
| Bogotá | 382.20 | 130.91 | 131 |
| México | 378.12 | 124.68 | 125 |
| Medellín | 373.53 | 99.18 | 44 |

Santiago presenta el promedio más alto de desechos, mientras que Medellín el más bajo.

**Promedio de Toneladas por Mes:**

| Mes | Media | n |
|-----|-------|---|
| Agosto | 424.13 | 16 |
| Septiembre | 419.47 | 4 |
| Enero | 406.07 | 38 |
| Febrero | 402.28 | 79 |
| Marzo | 393.07 | 85 |
| Abril | 392.19 | 113 |
| Junio | 390.74 | 64 |
| Octubre | 390.19 | 3 |
| Diciembre | 382.35 | 69 |
| Julio | 371.24 | 29 |

No se observan diferencias estacionales marcadas, aunque Julio presenta el promedio más bajo.

### Matriz de correlación

![Matriz de correlación](img/grafica_5.png)

**Correlaciones con Toneladas (variable dependiente):**

| Variable | Correlación |
|----------|-------------|
| Ventas | **0.863** (fuerte positiva) |
| Produccion | 0.089 |
| Agua | 0.053 |
| Nailon | 0.040 |
| Poliéster | 0.035 |
| Energia | 0.015 |
| Inventario | -0.038 |
| Proteccion | -0.069 |
| Algodón | -0.084 |
| Productividad | **-0.232** (moderada negativa) |

Ventas es la variable con mayor poder predictivo. Productividad tiene una relación negativa moderada: a mayor productividad energética, menores desechos.

### Dispersión: Variables continuas vs Toneladas

![Dispersión vs Toneladas](img/grafica_6.png)

Se confirma visualmente la fuerte relación lineal entre Ventas y Toneladas. Las demás variables muestran relaciones más débiles.

---

## 3. Estimación Matricial del Modelo (Punto 2)

Se estimó el modelo de regresión lineal múltiple por MCO usando solo variables continuas:

$$\hat{\beta} = (X^TX)^{-1}X^TY$$

### Coeficientes estimados

| Variable | Coeficiente | Error Estándar | t | p-valor | Significativo (α=0.05) |
|----------|-------------|----------------|---|---------|------------------------|
| Intercepto | 156.437 | 23.526 | 6.650 | <0.001 | Sí |
| Ventas | 3.941 | 0.081 | 48.572 | <0.001 | Sí |
| Produccion | 2.038 | 1.061 | 1.921 | 0.055 | No |
| Inventario | -46.589 | 23.229 | -2.006 | 0.045 | Sí |
| Algodón | -118.434 | 29.523 | -4.012 | <0.001 | Sí |
| Nailon | 38.438 | 52.204 | 0.736 | 0.462 | No |
| Poliéster | 3.802 | 58.635 | 0.065 | 0.948 | No |
| Energia | 0.646 | 0.551 | 1.172 | 0.242 | No |
| Productividad | -11.559 | 0.741 | -15.591 | <0.001 | Sí |
| Proteccion | -6.563 | 1.930 | -3.401 | <0.001 | Sí |
| Agua | 0.093 | 0.091 | 1.016 | 0.310 | No |

**Variables significativas:** Ventas, Inventario, Algodón, Productividad y Protección.

### Interpretación de coeficientes clave

- **Ventas (+3.94):** Por cada unidad adicional en ventas, los desechos aumentan ~3.94 toneladas.
- **Algodón (-118.43):** Un aumento de 1 punto porcentual en algodón orgánico reduce los desechos en ~1.18 toneladas.
- **Productividad (-11.56):** Mayor productividad energética reduce significativamente los desechos.
- **Protección (-6.56):** Mayor inversión en protección ambiental reduce los desechos.
- **Inventario (-46.59):** Mayor porcentaje de inventario se asocia con menos desechos (efecto contraintuitivo que puede reflejar mejor planificación).

### Tabla ANOVA y Prueba F

| Fuente | GL | SS | MS | F | p-valor |
|--------|----|----|----|----|---------|
| Regresión | 10 | 6,682,165.72 | 668,216.57 | 260.46 | <0.001 |
| Error | 489 | 1,254,520.12 | 2,565.48 | | |
| Total | 499 | 7,936,685.84 | | | |

- **R² = 0.8419** → El modelo explica el 84.19% de la variabilidad.
- **R² ajustado = 0.8387**
- **F = 260.46, p < 0.001** → El modelo es globalmente significativo.

---

## 4. Verificación con statsmodels (Punto 3)

Los coeficientes calculados matricialmente coinciden exactamente con los de statsmodels (diferencias del orden de 10⁻¹¹ a 10⁻¹⁴), validando la implementación matricial.

---

## 5. Verificación de Supuestos (Punto 4)

### Gráficos de diagnóstico de residuales

![Diagnóstico de residuales](img/grafica_7.png)

### 5.1 Normalidad de residuales

| Test | Estadístico | p-valor | Conclusión |
|------|-------------|---------|------------|
| Shapiro-Wilk | W = 0.9955 | 0.1636 | No se rechaza H₀: residuales normales |
| Jarque-Bera | JB = 3.578 | 0.1672 | No se rechaza H₀: residuales normales |

**✓ Supuesto cumplido.** Los residuales siguen una distribución normal.

### 5.2 Homocedasticidad

| Test | Estadístico | p-valor | Conclusión |
|------|-------------|---------|------------|
| Breusch-Pagan | LM = 12.41 | 0.2586 | No se rechaza H₀: homocedasticidad |

**✓ Supuesto cumplido.** La varianza de los errores es constante.

### 5.3 No autocorrelación

| Test | Estadístico | Conclusión |
|------|-------------|------------|
| Durbin-Watson | DW = 1.983 | No hay evidencia de autocorrelación |

**✓ Supuesto cumplido.** DW ≈ 2 indica ausencia de autocorrelación.

### 5.4 Multicolinealidad (VIF)

| Variable | VIF | Diagnóstico |
|----------|-----|-------------|
| Nailon | 166.18 | Severo |
| Poliéster | 171.83 | Severo |
| Agua | 18.27 | Severo |
| Proteccion | 17.12 | Severo |
| Productividad | 14.14 | Severo |
| Produccion | 12.88 | Severo |
| Ventas | 12.38 | Severo |
| Energia | 11.98 | Severo |
| Algodón | 10.28 | Severo |
| Inventario | 7.23 | Moderado |

**⚠ Supuesto NO cumplido.** Se detecta multicolinealidad severa, especialmente entre Nailon y Poliéster (VIF > 160). Esto infla los errores estándar y hace que variables como Nailon, Poliéster, Energía y Agua no resulten significativas individualmente, aunque el modelo global sí lo es.

**Corrección aplicada:** Se estimó el modelo con errores estándar robustos (HC3), que mantiene los mismos coeficientes pero ajusta los errores estándar. Las conclusiones de significancia se mantienen esencialmente iguales.

### Residuales vs cada variable independiente

![Residuales vs variables](img/grafica_8.png)

No se observan patrones sistemáticos en los gráficos de residuales vs variables individuales, lo que respalda la linealidad del modelo.

---

## 6. Modelo Completo con Variables Categóricas (Punto 5)

Al incorporar Ciudad y Mes como variables dummy (con `drop_first=True`, categoría base: Bogotá para Ciudad, Abril para Mes):

| Métrica | Solo continuas | Con categóricas |
|---------|----------------|-----------------|
| R² | 0.8419 | **0.8625** |
| R² ajustado | 0.8387 | **0.8562** |

La inclusión de las variables categóricas mejora el R² ajustado en ~1.8 puntos porcentuales.

**Variables categóricas significativas:**
- **Ciudad_Santiago (+24.83, p < 0.001):** Santiago genera ~25 toneladas más de desechos que Bogotá.
- **Ciudad_Medellin (-19.62, p = 0.021):** Medellín genera ~20 toneladas menos que Bogotá.
- **Mes_Julio (-24.10, p = 0.017):** En julio se generan ~24 toneladas menos que en abril.

---

## 7. Contrastes de Hipótesis (Punto 6)

### 6a. ¿El efecto de la productividad energética es mayor al de la protección?

| | Valor |
|---|---|
| H₀ | β_Productividad − β_Protección = 0 |
| H₁ | β_Productividad − β_Protección > 0 |
| Estimador (c'β) | -6.297 |
| Error estándar | 2.009 |
| t | -3.135 |
| p-valor (unilateral) | 0.999 |

**Conclusión:** No se rechaza H₀. No hay evidencia de que el efecto de la productividad sea mayor al de la protección. De hecho, el estimador es negativo, lo que sugiere que la productividad tiene un efecto más fuerte en magnitud absoluta (β = -12.09) que la protección (β = -5.79), pero ambos son negativos (reducen desechos), y la diferencia β_Prod − β_Prot es negativa, no positiva.

### 6b. ¿Los desechos en Santiago son mayores que en Bogotá?

| | Valor |
|---|---|
| H₀ | Desechos Santiago ≤ Desechos Bogotá |
| H₁ | Desechos Santiago > Desechos Bogotá |
| Estimador | 24.835 |
| Error estándar | 5.518 |
| t | 4.501 |
| p-valor (unilateral) | 0.000004 |

**Conclusión:** Se rechaza H₀. Los desechos en Santiago son significativamente mayores que en Bogotá (~25 toneladas más, controlando por las demás variables).

### 6c-1. ¿El efecto del algodón orgánico es diferente al del poliéster?

| | Valor |
|---|---|
| H₀ | β_Algodón − β_Poliéster = 0 |
| H₁ | β_Algodón − β_Poliéster ≠ 0 |
| Estimador | -110.761 |
| Error estándar | 62.561 |
| t | -1.770 |
| p-valor (bilateral) | 0.077 |

**Conclusión:** No se rechaza H₀ al 5%. No hay evidencia estadística suficiente para afirmar que el efecto del algodón sea diferente al del poliéster, aunque el p-valor es cercano al umbral (0.077), sugiriendo una tendencia.

### 6c-2. ¿El efecto de las ventas es el doble que el del agua?

| | Valor |
|---|---|
| H₀ | β_Ventas − 2·β_Agua = 0 |
| H₁ | β_Ventas − 2·β_Agua ≠ 0 |
| Estimador | 3.828 |
| Error estándar | 0.193 |
| t | 19.814 |
| p-valor (bilateral) | <0.001 |

**Conclusión:** Se rechaza H₀. El efecto de las ventas NO es el doble que el del agua. Las ventas tienen un impacto mucho mayor que el doble del agua sobre los desechos.

---

## 8. Propuesta de Minimización de Desechos — Julio en Bogotá (Punto 7)

Estrategia: usar percentil 25 para variables con coeficiente positivo (reducir) y percentil 75 para variables con coeficiente negativo (aumentar).

| Variable | Valor propuesto | Coeficiente | Estrategia |
|----------|----------------|-------------|------------|
| Ventas | 80.84 | +3.93 | P25 (minimizar) |
| Produccion | 6.41 | +2.58 | P25 (minimizar) |
| Inventario | 0.31 | -22.93 | P75 (maximizar) |
| Algodón | 0.30 | -108.43 | P75 (maximizar) |
| Nailon | 0.42 | +43.29 | P25 (minimizar) |
| Poliéster | 0.39 | +2.33 | P25 (minimizar) |
| Energia | 11.96 | +0.46 | P25 (minimizar) |
| Productividad | 14.04 | -12.09 | P75 (maximizar) |
| Proteccion | 5.82 | -5.79 | P75 (maximizar) |
| Agua | 95.67 | +0.05 | P25 (minimizar) |

**Resultado:**
- Predicción de desechos: **245.23 toneladas**
- Media actual del dataset: 393.46 toneladas
- Reducción estimada: **148.23 toneladas (37.7%)**
- Intervalo de confianza (95%): **[224.47, 265.98] toneladas**

---

## 9. Recomendaciones (Punto 8)

### Recomendación 1: Priorizar el uso de algodón orgánico sobre fibras sintéticas
El algodón orgánico tiene un coeficiente fuertemente negativo (β = -108.43), lo que significa que aumentar su proporción reduce significativamente los desechos. Se recomienda a las marcas transicionar hacia materias primas orgánicas y reducir la dependencia de fibras sintéticas como nailon y poliéster.

### Recomendación 2: Invertir en productividad energética
La productividad energética es la segunda variable más influyente (β = -12.09). Las marcas deben invertir en tecnologías más eficientes que permitan producir más con menos consumo energético, reduciendo desechos y costos operativos simultáneamente.

### Recomendación 3: Fortalecer la protección ambiental
La inversión en protección ambiental tiene un efecto negativo significativo sobre los desechos (β = -5.79). Implementar programas de gestión ambiental y certificaciones de sostenibilidad contribuye directamente a la reducción de desechos textiles.

### Recomendación adicional: Atención especial a Santiago de Chile
Santiago genera significativamente más desechos que las demás ciudades (~25 toneladas más que Bogotá). Se recomienda focalizar las estrategias de reducción en esta ciudad, investigando los factores locales que contribuyen a esta diferencia.

---

## Resumen de Hallazgos Clave

1. El modelo explica el **84.2%** de la variabilidad en desechos textiles (86.3% con categóricas).
2. **Ventas** es el predictor dominante (r = 0.86), seguido de **Productividad** y **Algodón**.
3. Los supuestos de normalidad, homocedasticidad y no autocorrelación se cumplen. La multicolinealidad es un problema que afecta la significancia individual de algunas variables.
4. Santiago genera más desechos que Bogotá (p < 0.001). Julio es el mes con menor generación.
5. Con la propuesta de optimización, se estima una reducción del **37.7%** en desechos para Julio en Bogotá.
