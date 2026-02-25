#modelo matricial
#matriz x con columna de 1 (intercepto)
x = data.matrix(matrix)
#vector de la variable dependiente
y = data.matrix(data$Inversion)

# multiplicación XTY
xty = t(x)%*%y
xty
# inversa de XTX
xtx_inv = solve(t(x)%*%x)

#cálculo de los coeficientes - mínimos cuadraros ordinarios
beta = xtx_inv%*%xty 
beta 


#correr modelo regresión con dos variables 
fits = lm(Inversion~Presion+Edad, data = data)
summary(fits)


#correr modelo regresión múltiple----------------------------------------------------------------------------------
fit = lm(Inversion~., data = data)
summary(fit)




## Supuestos --------------------------------------------------------------------
#Para modelo inicial 

#librerías 
install.packages("ggplot2")
install.packages("GGally")
install.packages("car")
install.packages("lmtest")
library(lmtest)
library(GGally)
library(car)

#probar supuestos 
#residuales 
plot(residuals(fit))


#heteroscedasticidad

bptest(fit)

#autocorrelación
dwtest(fit)

#se prueba multicolinealidad 

ggpairs(data)
vif(fit)


#contraste---------------------------------------------------------------------------------------------- 
#prueba hip?tesis t , combinaci?n lineal de betas
betas = coefficients(fit)
c = data.matrix(c(0,-2,0,1,0,0,0,0,0,0,0,0))

estimador = t(c)%*%betas
estimador
varianza = t(c)%*%data.matrix(vcov(fit))%*%c
varianza
estimador/(varianza^(0.5))

qt(0.95,114)


#categóricas------------------------------------------------------------------------ 
datos$Yoga = as.factor(datos$Yoga)
datos$Ciudad = as.factor(datosl$Ciudad)


#intervalo de confianza para la media -------------------------------------

b= coefficients(fit)
#valores de prueba para las variables independientes 
x = data.matrix(c(1,100,30,200,4,0,1,0,0,0,0))

estimador = t(x)%*%b
estimador
varianza = t(x)%*%data.matrix(vcov(fit3))%*%x
varianza

inf = estimador - (1.98*(varianza^(0.5)))
inf
sup = estimador + (1.98*(varianza^(0.5)))
sup


