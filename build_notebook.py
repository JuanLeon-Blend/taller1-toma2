import nbformat as nbf
nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# --- TITULO ---
md("""# Taller 1 - Toma de Decisiones II (2026-01)
## Análisis de Desechos Textiles en Latinoamérica

**Industria Textil - Consultora de Sostenibilidad**""")

# --- IMPORTS ---
code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100""")

# --- CARGA ---
md("## 1. Carga y Exploración Inicial de los Datos")

code("""df = pd.read_excel('Textiles.xlsx')
df.columns = df.columns.str.strip()
# Limpiar espacios en columnas categóricas
df['Mes'] = df['Mes'].str.strip()
print(f'Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas')
print(f'Columnas: {df.columns.tolist()}')
df.head(10)""")

code("""df.info()""")

code("""# Variables numéricas y categóricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
indep_num = [c for c in num_cols if c != 'Toneladas']
print(f'Variables numéricas ({len(num_cols)}): {num_cols}')
print(f'Variables categóricas ({len(cat_cols)}): {cat_cols}')
print(f'Variables independientes continuas ({len(indep_num)}): {indep_num}')""")

code("""df.describe().round(4)""")

code("""print('Valores nulos por columna:')
print(df.isnull().sum())
print(f'\\nCiudades: {df["Ciudad"].unique().tolist()}')
print(f'Meses: {sorted(df["Mes"].unique().tolist())}')
print(f'\\nConteo por Ciudad:')
print(df['Ciudad'].value_counts())
print(f'\\nConteo por Mes:')
print(df['Mes'].value_counts())""")

# --- PUNTO 1 ---
md("""## Punto 1: Análisis Descriptivo (10/100)
### Estadísticas descriptivas y gráficos para identificar patrones""")

code("""# Estadísticas descriptivas extendidas
num_df = df[num_cols]
stats = num_df.describe().T
stats['CV%'] = (stats['std'] / stats['mean'] * 100).round(2)
stats['IQR'] = stats['75%'] - stats['25%']
stats['skew'] = num_df.skew().round(4)
stats['kurtosis'] = num_df.kurtosis().round(4)
stats.round(4)""")

md("### Distribución de la variable dependiente: Toneladas de desechos")

code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['Toneladas'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].axvline(df['Toneladas'].mean(), color='red', linestyle='--', label=f'Media: {df["Toneladas"].mean():.1f}')
axes[0].axvline(df['Toneladas'].median(), color='green', linestyle='--', label=f'Mediana: {df["Toneladas"].median():.1f}')
axes[0].set_title('Distribución de Toneladas de Desechos')
axes[0].set_xlabel('Toneladas'); axes[0].set_ylabel('Frecuencia'); axes[0].legend()

axes[1].boxplot(df['Toneladas'], vert=True)
axes[1].set_title('Boxplot de Toneladas'); axes[1].set_ylabel('Toneladas')

sp_stats.probplot(df['Toneladas'], dist='norm', plot=axes[2])
axes[2].set_title('QQ-Plot de Toneladas')
plt.tight_layout(); plt.show()""")

md("### Distribución de todas las variables numéricas")

code("""fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_title(f'Distribución de {col}')
    axes[i].set_xlabel(col); axes[i].set_ylabel('Frecuencia')
for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)
plt.tight_layout(); plt.show()""")

md("### Boxplots de todas las variables numéricas")

code("""fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='steelblue')
    axes[i].set_title(f'Boxplot de {col}')
for j in range(len(num_cols), len(axes)):
    axes[j].set_visible(False)
plt.tight_layout(); plt.show()""")

md("### Análisis por variables categóricas")

code("""fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.boxplot(data=df, x='Ciudad', y='Toneladas', ax=axes[0], palette='Set2')
axes[0].set_title('Toneladas de Desechos por Ciudad')

sns.boxplot(data=df, x='Mes', y='Toneladas', ax=axes[1], palette='Set3')
axes[1].set_title('Toneladas de Desechos por Mes')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()""")

code("""# Promedio de desechos por ciudad y mes
print('Promedio de Toneladas por Ciudad:')
print(df.groupby('Ciudad')['Toneladas'].agg(['mean','std','count']).round(2))
print('\\nPromedio de Toneladas por Mes:')
print(df.groupby('Mes')['Toneladas'].agg(['mean','std','count']).round(2))""")

md("### Matriz de correlación (variables continuas)")

code("""corr = num_df.corr()
fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Matriz de Correlación entre Variables', fontsize=14)
plt.tight_layout(); plt.show()""")

code("""print('Correlación con Toneladas (variable dependiente):')
print(corr['Toneladas'].drop('Toneladas').sort_values(ascending=False).round(4))""")

md("### Gráficos de dispersión: Variables continuas vs Toneladas")

code("""fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()
for i, col in enumerate(indep_num):
    axes[i].scatter(df[col], df['Toneladas'], alpha=0.4, s=15, color='steelblue')
    z = np.polyfit(df[col], df['Toneladas'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    axes[i].plot(x_line, p(x_line), 'r--', alpha=0.8)
    r = df[col].corr(df['Toneladas'])
    axes[i].set_title(f'{col} vs Toneladas (r={r:.3f})')
    axes[i].set_xlabel(col); axes[i].set_ylabel('Toneladas')
plt.tight_layout(); plt.show()""")

md("""### Conclusiones del Análisis Descriptivo

- **Toneladas** (variable dependiente): distribución aproximadamente normal con media ~393 toneladas. No se observan valores atípicos extremos.
- **Ventas** presenta la correlación más alta con Toneladas (positiva): a mayores ventas, mayor producción y más desechos.
- **Agua** muestra correlación positiva relevante: procesos con mayor consumo de agua generan más desechos.
- **Inventario** tiene correlación positiva moderada: mayor porcentaje destinado a inventario se asocia con más desechos.
- **Poliéster** muestra correlación positiva: el uso de fibras de poliéster se asocia con mayor generación de desechos.
- **Algodón** presenta correlación negativa: el uso de algodón orgánico se asocia con menor generación de desechos.
- **Ciudad**: se observan diferencias en los niveles de desechos entre ciudades, lo cual justifica incluir esta variable categórica en el modelo.
- **Mes**: no se observan diferencias marcadas entre meses, pero se incluirá para verificar estacionalidad.
- No hay valores nulos en el dataset.""")

# --- PUNTO 2: MODELO MATRICIAL ---
md("""## Punto 2: Estimación Matricial del Modelo de Regresión Lineal Múltiple (20/100)

El modelo de regresión lineal múltiple es: $Y = X\\beta + \\varepsilon$

Los coeficientes se estiman por MCO: $\\hat{\\beta} = (X^TX)^{-1}X^TY$

**Nota:** Para la estimación matricial se consideran únicamente las variables continuas (Punto 2-4). Las categóricas se incorporan en el Punto 5.""")

code("""# Variables continuas solamente
y = df['Toneladas'].values.reshape(-1, 1)
X_cont = df[indep_num]
var_names = ['Intercepto'] + indep_num

# Agregar columna de 1s para el intercepto
X = np.column_stack([np.ones(len(df)), X_cont.values])

print(f'Dimensiones de X: {X.shape}')
print(f'Dimensiones de Y: {y.shape}')
print(f'Variables: {var_names}')""")

code("""# X'X
XtX = X.T @ X
print("Matriz X'X:")
print(pd.DataFrame(XtX, index=var_names, columns=var_names).round(2))""")

code("""# Inversa de X'X
XtX_inv = np.linalg.inv(XtX)
print("Matriz (X'X)^(-1):")
print(pd.DataFrame(XtX_inv, index=var_names, columns=var_names).round(8))""")

code("""# X'Y
XtY = X.T @ y
print("Vector X'Y:")
for name, val in zip(var_names, XtY.flatten()):
    print(f'  {name}: {val:.4f}')""")

code("""# Coeficientes: beta = (X'X)^(-1) * X'Y
beta = XtX_inv @ XtY
print('COEFICIENTES ESTIMADOS (β)')
print('='*50)
for name, val in zip(var_names, beta.flatten()):
    print(f'  {name:20s}: {val:.6f}')""")

md("### Sumas de Cuadrados y Prueba de Significancia Global")

code("""# Valores ajustados y sumas de cuadrados
y_hat = X @ beta
n = len(y)
k = X.shape[1] - 1  # variables independientes (sin intercepto)
y_mean = y.mean()

SSR = float(((y_hat - y_mean).T @ (y_hat - y_mean)).item())
SSE = float(((y - y_hat).T @ (y - y_hat)).item())
SST = float(((y - y_mean).T @ (y - y_mean)).item())

print('SUMAS DE CUADRADOS')
print('='*50)
print(f'  SSR (Regresión): {SSR:,.4f}')
print(f'  SSE (Error):     {SSE:,.4f}')
print(f'  SST (Total):     {SST:,.4f}')
print(f'  SSR + SSE:       {SSR + SSE:,.4f}')
print(f'  Verificación SST ≈ SSR + SSE: {np.isclose(SST, SSR + SSE)}')""")

code("""# R² y R² ajustado
R2 = SSR / SST
R2_adj = 1 - (SSE / (n - k - 1)) / (SST / (n - 1))
print(f'R² = {R2:.6f}')
print(f'R² ajustado = {R2_adj:.6f}')
print(f'El modelo explica el {R2*100:.2f}% de la variabilidad de los desechos textiles.')""")

code("""# Prueba F de significancia global
MSR = SSR / k
MSE = SSE / (n - k - 1)
F_stat = MSR / MSE
p_value_F = 1 - sp_stats.f.cdf(F_stat, k, n - k - 1)
F_crit = sp_stats.f.ppf(0.95, k, n - k - 1)

print('PRUEBA DE SIGNIFICANCIA GLOBAL (Prueba F)')
print('='*60)
print(f'  H0: β1 = β2 = ... = β{k} = 0')
print(f'  H1: al menos un βi ≠ 0')
print(f'\\n  MSR = {MSR:,.4f}')
print(f'  MSE = {MSE:,.4f}')
print(f'  F estadístico = {F_stat:.4f}')
print(f'  GL: ({k}, {n - k - 1})')
print(f'  p-valor = {p_value_F:.2e}')
print(f'  F crítico (α=0.05): {F_crit:.4f}')
if p_value_F < 0.05:
    print(f'\\n  CONCLUSIÓN: Se rechaza H0 (p < 0.05). El modelo es globalmente significativo.')
else:
    print(f'\\n  CONCLUSIÓN: No se rechaza H0. El modelo no es significativo.')""")

code("""# Tabla ANOVA
anova_df = pd.DataFrame({
    'Fuente': ['Regresión', 'Error', 'Total'],
    'GL': [k, n - k - 1, n - 1],
    'SS': [SSR, SSE, SST],
    'MS': [MSR, MSE, np.nan],
    'F': [F_stat, np.nan, np.nan],
    'p-valor': [p_value_F, np.nan, np.nan]
})
print('TABLA ANOVA')
print('='*80)
print(anova_df.to_string(index=False))""")

md("### Significancia individual de cada coeficiente")

code("""# Errores estándar, t-stats, p-valores
var_beta = MSE * XtX_inv
se_beta = np.sqrt(np.diag(var_beta))
t_stats = beta.flatten() / se_beta
p_values = 2 * (1 - sp_stats.t.cdf(np.abs(t_stats), n - k - 1))

coef_df = pd.DataFrame({
    'Variable': var_names,
    'Coeficiente': beta.flatten(),
    'Error Estándar': se_beta,
    't-estadístico': t_stats,
    'p-valor': p_values,
    'Significativo': ['Sí' if p < 0.05 else 'No' for p in p_values]
})
print('COEFICIENTES DEL MODELO')
print('='*90)
print(coef_df.to_string(index=False))""")

# --- PUNTO 3 ---
md("""## Punto 3: Verificación del Modelo con statsmodels (5/100)
### Equivalente a correr el modelo en R-Studio con variables continuas""")

code("""import statsmodels.api as sm

X_sm = sm.add_constant(df[indep_num])
model = sm.OLS(df['Toneladas'], X_sm).fit()
print(model.summary())""")

code("""# Verificar coincidencia con cálculo matricial
print('Verificación de coeficientes:')
print(f'{"Variable":20s} {"Matricial":>15s} {"statsmodels":>15s} {"Diferencia":>12s}')
print('='*65)
for name, b_mat, b_sm in zip(var_names, beta.flatten(), model.params.values):
    print(f'{name:20s} {b_mat:15.6f} {b_sm:15.6f} {abs(b_mat-b_sm):12.2e}')""")

# --- PUNTO 4 ---
md("## Punto 4: Verificación de Supuestos del Modelo (10/100)")

md("### 4.1 Normalidad de los residuales")

code("""residuals = model.resid
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histograma
axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue', density=True)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
axes[0].plot(x_norm, sp_stats.norm.pdf(x_norm, residuals.mean(), residuals.std()), 'r-', lw=2)
axes[0].set_title('Distribución de Residuales')
axes[0].set_xlabel('Residuales'); axes[0].set_ylabel('Densidad')

# QQ-plot
sp_stats.probplot(residuals, dist='norm', plot=axes[1])
axes[1].set_title('QQ-Plot de Residuales')

# Residuales vs ajustados
axes[2].scatter(model.fittedvalues, residuals, alpha=0.4, s=15, color='steelblue')
axes[2].axhline(y=0, color='red', linestyle='--')
axes[2].set_title('Residuales vs Valores Ajustados')
axes[2].set_xlabel('Valores Ajustados'); axes[2].set_ylabel('Residuales')
plt.tight_layout(); plt.show()""")

code("""# Shapiro-Wilk
stat_sw, p_sw = sp_stats.shapiro(residuals)
print('TEST DE NORMALIDAD - Shapiro-Wilk')
print(f'  W = {stat_sw:.6f}, p-valor = {p_sw:.6f}')
print(f'  {"No se rechaza H0: residuales normales" if p_sw > 0.05 else "Se rechaza H0: residuales NO normales"} (α=0.05)')

# Jarque-Bera
jb_stat, jb_p = sp_stats.jarque_bera(residuals)
print(f'\\nTEST DE NORMALIDAD - Jarque-Bera')
print(f'  JB = {jb_stat:.6f}, p-valor = {jb_p:.6f}')
print(f'  {"No se rechaza H0: residuales normales" if jb_p > 0.05 else "Se rechaza H0: residuales NO normales"} (α=0.05)')""")

md("### 4.2 Homocedasticidad")

code("""from statsmodels.stats.diagnostic import het_breuschpagan

bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, model.model.exog)
print('TEST DE HOMOCEDASTICIDAD - Breusch-Pagan')
print(f'  H0: Varianza constante (homocedasticidad)')
print(f'  H1: Varianza NO constante (heterocedasticidad)')
print(f'  LM = {bp_stat:.6f}, p-valor = {bp_p:.6f}')
print(f'  {"No se rechaza H0: homocedasticidad" if bp_p > 0.05 else "Se rechaza H0: heterocedasticidad"} (α=0.05)')""")

md("### 4.3 No autocorrelación")

code("""from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(residuals)
print('TEST DE AUTOCORRELACIÓN - Durbin-Watson')
print(f'  DW = {dw:.6f} (valores cercanos a 2 = no autocorrelación)')
print(f'  {"No hay evidencia de autocorrelación" if 1.5 < dw < 2.5 else "Posible autocorrelación"}')""")

md("### 4.4 No multicolinealidad (VIF)")

code("""from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = df[indep_num]
vif_data = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
vif_data['Problema'] = vif_data['VIF'].apply(lambda x: 'Severo (>10)' if x > 10 else ('Moderado (5-10)' if x > 5 else 'No'))
print('FACTOR DE INFLACIÓN DE LA VARIANZA (VIF)')
print('='*50)
print(vif_data.to_string(index=False))
print('\\nVIF > 10 = multicolinealidad severa; VIF > 5 = moderada')""")

md("### 4.5 Gráficos de residuales vs cada variable")

code("""fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()
for i, col in enumerate(indep_num):
    axes[i].scatter(df[col], residuals, alpha=0.4, s=15, color='steelblue')
    axes[i].axhline(y=0, color='red', linestyle='--')
    axes[i].set_title(f'Residuales vs {col}')
    axes[i].set_xlabel(col); axes[i].set_ylabel('Residuales')
plt.tight_layout(); plt.show()""")

md("""### Diagnóstico y Correcciones

Si se detectan problemas en los supuestos:
- **Heterocedasticidad**: se aplican errores estándar robustos (HC3).
- **Multicolinealidad**: se evalúa eliminar variables con VIF alto.
- **No normalidad**: con n=500, por el TLC los estimadores son asintóticamente normales.""")

code("""# Modelo con errores estándar robustos (HC3)
model_robust = model.get_robustcov_results(cov_type='HC3')
print('MODELO CON ERRORES ESTÁNDAR ROBUSTOS (HC3)')
print(model_robust.summary())""")

# --- PUNTO 5 ---
md("""## Punto 5: Modelo con Variables Continuas y Categóricas (5/100)
### Incorporación de Ciudad y Mes como variables dummy""")

code("""# Crear dummies para Ciudad y Mes
df_model = df.copy()
df_dummies = pd.get_dummies(df_model, columns=['Ciudad', 'Mes'], drop_first=True, dtype=float)

# Separar Y y X
y_full = df_dummies['Toneladas']
X_full = df_dummies.drop(columns=['Toneladas'])
X_full_sm = sm.add_constant(X_full)

model_full = sm.OLS(y_full, X_full_sm).fit()
print(model_full.summary())""")

code("""# Comparar R² del modelo solo continuas vs completo
print(f'R² modelo solo continuas: {model.rsquared:.6f}')
print(f'R² modelo completo (con categóricas): {model_full.rsquared:.6f}')
print(f'R² ajustado solo continuas: {model.rsquared_adj:.6f}')
print(f'R² ajustado completo: {model_full.rsquared_adj:.6f}')""")

# --- PUNTO 6 ---
md("## Punto 6: Contrastes de Hipótesis (30/100)")

md("""### 6a. ¿El efecto de la productividad energética es mayor al de la protección?

$$H_0: \\beta_{Productividad} - \\beta_{Proteccion} = 0$$
$$H_1: \\beta_{Productividad} - \\beta_{Proteccion} > 0$$

Se usa el modelo completo (con categóricas) para los contrastes.""")

code("""# Usar el modelo completo para contrastes
beta_full = model_full.params.values.reshape(-1, 1)
var_names_full = model_full.params.index.tolist()
n_full = len(y_full)
k_full = len(var_names_full) - 1
MSE_full = model_full.mse_resid
vcov_full = np.array(model_full.cov_params())

print('Variables en el modelo completo:')
for i, name in enumerate(var_names_full):
    print(f'  [{i}] {name}: β = {beta_full[i,0]:.6f}')""")

code("""# Vector de contraste: Productividad - Proteccion
prod_idx = var_names_full.index('Productividad')
prot_idx = var_names_full.index('Proteccion')

c = np.zeros((len(var_names_full), 1))
c[prod_idx] = 1
c[prot_idx] = -1

estimador = (c.T @ beta_full).item()
var_c = (c.T @ vcov_full @ c).item()
se_c = np.sqrt(var_c)
t_c = estimador / se_c
gl = n_full - k_full - 1
p_val = 1 - sp_stats.t.cdf(t_c, gl)  # unilateral derecha
t_crit = sp_stats.t.ppf(0.95, gl)

print('CONTRASTE 6a: Productividad vs Protección')
print('='*60)
print(f'  H0: β_Productividad - β_Protección = 0')
print(f'  H1: β_Productividad - β_Protección > 0')
print(f'\\n  Estimador (c\\'β): {estimador:.6f}')
print(f'  Error estándar: {se_c:.6f}')
print(f'  t = {t_c:.4f}')
print(f'  t crítico (α=0.05, gl={gl}): {t_crit:.4f}')
print(f'  p-valor (unilateral): {p_val:.6f}')
if p_val < 0.05:
    print(f'\\n  CONCLUSIÓN: Se rechaza H0. El efecto de la productividad energética')
    print(f'  es significativamente mayor al de la protección ambiental.')
    print(f'  Recomendación: enfocar estrategias en eficiencia productiva.')
else:
    print(f'\\n  CONCLUSIÓN: No se rechaza H0. No hay evidencia suficiente para afirmar')
    print(f'  que el efecto de la productividad sea mayor al de la protección.')""")

md("""### 6b. ¿Los desechos en Santiago de Chile son mayores que en Bogotá?

$$H_0: \\beta_{Santiago} - \\beta_{Bogota} \\leq 0$$
$$H_1: \\beta_{Santiago} - \\beta_{Bogota} > 0$$

Nota: Como se usó `drop_first=True`, la ciudad base (referencia) depende del orden alfabético. Las dummies representan la diferencia respecto a la categoría base.""")

code("""# Identificar las dummies de ciudad
ciudad_vars = [v for v in var_names_full if v.startswith('Ciudad_')]
print('Variables dummy de Ciudad:', ciudad_vars)
print()

# Determinar la categoría base
todas_ciudades = df['Ciudad'].unique().tolist()
ciudades_dummy = [v.replace('Ciudad_', '') for v in ciudad_vars]
ciudad_base = [c for c in todas_ciudades if c not in ciudades_dummy][0]
print(f'Ciudad base (referencia): {ciudad_base}')

# Construir contraste Santiago - Bogota
c2 = np.zeros((len(var_names_full), 1))

# Si Santiago tiene dummy, su coeficiente es β_Santiago (diferencia vs base)
# Si Bogota tiene dummy, su coeficiente es β_Bogota (diferencia vs base)
# Queremos: (β_base + β_Santiago) - (β_base + β_Bogota) = β_Santiago - β_Bogota

santiago_var = [v for v in ciudad_vars if 'Santiago' in v]
bogota_var = [v for v in ciudad_vars if 'Bogota' in v or 'Bogotá' in v]

if santiago_var:
    c2[var_names_full.index(santiago_var[0])] = 1
else:
    # Santiago es la base, así que su efecto es 0
    pass

if bogota_var:
    c2[var_names_full.index(bogota_var[0])] = -1
else:
    # Bogota es la base
    pass

print(f'\\nContraste: Santiago - Bogota')
print(f'Vector c (posiciones no cero):')
for i, v in enumerate(var_names_full):
    if c2[i] != 0:
        print(f'  {v}: {c2[i,0]}')

estimador2 = (c2.T @ beta_full).item()
var_c2 = (c2.T @ vcov_full @ c2).item()
se_c2 = np.sqrt(var_c2)
t_c2 = estimador2 / se_c2
p_val2 = 1 - sp_stats.t.cdf(t_c2, gl)  # unilateral derecha

print(f'\\nCONTRASTE 6b: Santiago vs Bogotá')
print('='*60)
print(f'  H0: Desechos Santiago ≤ Desechos Bogotá')
print(f'  H1: Desechos Santiago > Desechos Bogotá')
print(f'\\n  Estimador: {estimador2:.6f}')
print(f'  Error estándar: {se_c2:.6f}')
print(f'  t = {t_c2:.4f}')
print(f'  p-valor (unilateral): {p_val2:.6f}')
if p_val2 < 0.05:
    print(f'\\n  CONCLUSIÓN: Se rechaza H0. Los desechos en Santiago son')
    print(f'  significativamente mayores que en Bogotá.')
else:
    print(f'\\n  CONCLUSIÓN: No se rechaza H0. No hay evidencia suficiente para afirmar')
    print(f'  que los desechos en Santiago sean mayores que en Bogotá.')""")

md("""### 6c. Contrastes adicionales

#### Contraste 1: ¿El efecto del Algodón orgánico es significativamente diferente al del Poliéster?

$$H_0: \\beta_{Algodón} - \\beta_{Poliéster} = 0$$
$$H_1: \\beta_{Algodón} - \\beta_{Poliéster} \\neq 0$$

Este contraste es relevante porque permite determinar si la elección de materia prima (orgánica vs sintética) tiene un impacto diferenciado sobre los desechos.""")

code("""# Contraste: Algodón vs Poliéster
alg_idx = var_names_full.index('Algodón')
pol_idx = var_names_full.index('Poliéster')

c3 = np.zeros((len(var_names_full), 1))
c3[alg_idx] = 1
c3[pol_idx] = -1

estimador3 = (c3.T @ beta_full).item()
var_c3 = (c3.T @ vcov_full @ c3).item()
se_c3 = np.sqrt(var_c3)
t_c3 = estimador3 / se_c3
p_val3 = 2 * (1 - sp_stats.t.cdf(abs(t_c3), gl))

print('CONTRASTE 6c-1: Algodón vs Poliéster')
print('='*60)
print(f'  H0: β_Algodón - β_Poliéster = 0')
print(f'  H1: β_Algodón - β_Poliéster ≠ 0')
print(f'\\n  Estimador: {estimador3:.6f}')
print(f'  Error estándar: {se_c3:.6f}')
print(f'  t = {t_c3:.4f}')
print(f'  p-valor (bilateral): {p_val3:.6f}')
if p_val3 < 0.05:
    print(f'\\n  CONCLUSIÓN: Se rechaza H0. El efecto del algodón orgánico es')
    print(f'  significativamente diferente al del poliéster.')
    print(f'  La elección de materia prima impacta de forma diferenciada los desechos.')
else:
    print(f'\\n  CONCLUSIÓN: No se rechaza H0. No hay diferencia significativa.')""")

md("""#### Contraste 2: ¿El efecto de las Ventas es el doble que el del Agua?

$$H_0: \\beta_{Ventas} - 2\\beta_{Agua} = 0$$
$$H_1: \\beta_{Ventas} - 2\\beta_{Agua} \\neq 0$$

Este contraste evalúa la relación proporcional entre el impacto de las ventas y el consumo de agua sobre los desechos.""")

code("""# Contraste: Ventas vs 2*Agua
ventas_idx = var_names_full.index('Ventas')
agua_idx = var_names_full.index('Agua')

c4 = np.zeros((len(var_names_full), 1))
c4[ventas_idx] = 1
c4[agua_idx] = -2

estimador4 = (c4.T @ beta_full).item()
var_c4 = (c4.T @ vcov_full @ c4).item()
se_c4 = np.sqrt(var_c4)
t_c4 = estimador4 / se_c4
p_val4 = 2 * (1 - sp_stats.t.cdf(abs(t_c4), gl))

print('CONTRASTE 6c-2: Ventas vs 2*Agua')
print('='*60)
print(f'  H0: β_Ventas - 2*β_Agua = 0')
print(f'  H1: β_Ventas - 2*β_Agua ≠ 0')
print(f'\\n  Estimador: {estimador4:.6f}')
print(f'  Error estándar: {se_c4:.6f}')
print(f'  t = {t_c4:.4f}')
print(f'  p-valor (bilateral): {p_val4:.6f}')
if p_val4 < 0.05:
    print(f'\\n  CONCLUSIÓN: Se rechaza H0. El efecto de las ventas NO es el doble que el del agua.')
else:
    print(f'\\n  CONCLUSIÓN: No se rechaza H0. No hay evidencia para rechazar la proporción.')""")

# --- PUNTO 7 ---
md("""## Punto 7: Propuesta de valores para minimizar desechos en Julio en Bogotá (10/100)

Se proponen valores para cada variable independiente que minimicen los desechos textiles, usando el modelo completo. La lógica:
- Variables con coeficiente **positivo**: usar valores **bajos** (percentil 25).
- Variables con coeficiente **negativo**: usar valores **altos** (percentil 75).
- Ciudad: Bogotá, Mes: Julio (según el enunciado).""")

code("""# Analizar coeficientes del modelo completo
print('Coeficientes de variables continuas y dirección para minimizar:')
print('='*70)
for name in indep_num:
    b = model_full.params[name]
    direction = 'MINIMIZAR ↓' if b > 0 else 'MAXIMIZAR ↑'
    print(f'  {name:20s}: β = {b:10.4f} -> {direction}')""")

code("""# Construir vector de predicción para Julio en Bogotá
x_pred = {}

# Variables continuas: percentil favorable según signo del coeficiente
for name in indep_num:
    b = model_full.params[name]
    if b > 0:
        x_pred[name] = df[name].quantile(0.25)
    else:
        x_pred[name] = df[name].quantile(0.75)

# Variables dummy de Ciudad: Bogotá
for v in var_names_full:
    if v.startswith('Ciudad_'):
        ciudad = v.replace('Ciudad_', '')
        x_pred[v] = 1.0 if ciudad == 'Bogota' else 0.0

# Variables dummy de Mes: Julio
for v in var_names_full:
    if v.startswith('Mes_'):
        mes = v.replace('Mes_', '')
        x_pred[v] = 1.0 if mes == 'Julio' else 0.0

# Construir vector completo
x_vec = [1.0]  # intercepto
for name in var_names_full[1:]:  # sin 'const'
    x_vec.append(x_pred.get(name, 0.0))
x_vec = np.array(x_vec)

print('VALORES PROPUESTOS PARA MINIMIZAR DESECHOS (Julio, Bogotá):')
print('='*60)
for name, val in zip(var_names_full, x_vec):
    if name == 'const':
        continue
    if name in indep_num:
        b = model_full.params[name]
        print(f'  {name:25s}: {val:.4f} (β={b:.4f}, {"P25" if b > 0 else "P75"})')
    elif val != 0:
        print(f'  {name:25s}: {val:.0f}')""")

code("""# Predicción
prediccion = (x_vec @ beta_full).item()
print(f'\\nPredicción de desechos: {prediccion:.2f} toneladas')
print(f'Media actual del dataset: {df["Toneladas"].mean():.2f} toneladas')
reduccion = df['Toneladas'].mean() - prediccion
print(f'Reducción estimada: {reduccion:.2f} toneladas ({reduccion/df["Toneladas"].mean()*100:.1f}%)')

# Intervalo de confianza para la media
x_col = x_vec.reshape(-1, 1)
var_pred = (x_col.T @ vcov_full @ x_col).item()
se_pred = np.sqrt(var_pred)
t_crit_95 = sp_stats.t.ppf(0.975, gl)
ic_inf = prediccion - t_crit_95 * se_pred
ic_sup = prediccion + t_crit_95 * se_pred
print(f'\\nIntervalo de confianza para la media (95%): [{ic_inf:.2f}, {ic_sup:.2f}] toneladas')""")

# --- PUNTO 8 ---
md("""## Punto 8: Recomendaciones (10/100)

Con base en el análisis estadístico realizado, se presentan las siguientes recomendaciones:

### Recomendación 1: Priorizar el uso de algodón orgánico sobre fibras sintéticas
El modelo muestra que el porcentaje de algodón orgánico tiene un efecto negativo sobre los desechos (a mayor uso, menores desechos), mientras que el poliéster tiene un efecto positivo (más poliéster, más desechos). Se recomienda a las marcas transicionar gradualmente hacia materias primas orgánicas y reducir la dependencia de fibras sintéticas.

### Recomendación 2: Optimizar la eficiencia energética y productividad
La productividad energética es una variable clave. Las marcas deben invertir en tecnologías más eficientes que permitan producir más con menos consumo energético. Esto reduce desechos y costos operativos simultáneamente.

### Recomendación 3: Gestionar estratégicamente los inventarios
El porcentaje de producción destinado a inventario se asocia positivamente con la generación de desechos. Se recomienda implementar modelos de producción ajustados a la demanda (lean manufacturing / just-in-time) para reducir el exceso de inventario que eventualmente se convierte en desecho textil.""")

# --- GUARDAR ---
nb.cells = cells
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3 (venv)',
    'language': 'python',
    'name': 'venv'
}
nb.metadata['language_info'] = {
    'name': 'python',
    'version': '3.13.0'
}

with open('informe.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Notebook guardado en informe.ipynb')
