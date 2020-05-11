library(e1071) # libreria que contiene la Funcion de SVM
library(caTools) #libreria que contiene la funcion sample.split 
library (gplots) #libreria que permite visualizar graficos
library(readxl) #libreria que permite leer archives excel
library(ggplot2) #libreria que permite visualizar graficos mas especificos
library(caret) #permite realizar las divisiones de los datos (entranamiento y prueba)
library(dplyr) #accionescomunes que se realizan sobre un conjunto de datos
library(leaps) #contiene el metodo para encontrar las mejores caracteristicas

set.seed(12345) # establecer una semilla para que cada vez que se ejecute los valores no cambien

MasDatosNu <- read_excel("C:/Users/Sergio/Desktop/MasDatosNu.xlsx") # importamos en dataset en forma excel

str (MasDatosNu) #permite ver el tipo de dato por el que esta conformado del dataset

colnames(MasDatosNu)[26] <- "classification"  #linea que permite acceder a la columna clasifiaciï¿½n

MasDatosNu$classification <- factor(MasDatosNu$classification, levels = c("0", "1"), labels = c("notckd", "ckd")) # convertir la variable respuesta en factor


MasDatosNu[c(1:294),-1] # eliminamos la columna Id ya que es un dato irrelevante


MasDatosNulim = MasDatosNu[c(1:294),-1] # cargamos un nuevo dataset sin la columna ID

#se observa  que existe cierta variabilidad y diferencia de escala entre los valores
#mï¿½nimos y mï¿½ximos de los registros medidos, por lo que es conveniente, 
# el rescalar las variables numï¿½ricas a fin de que ninguna tenga 
#mayor influencia sobre otra como predictor al momento de construir los modelos de clasificaciï¿½n. 
#Entonces, usemos la funciï¿½n scale:

MasDatosNulim[, c(1:24)] <- scale(MasDatosNulim[, c(1:24)])

summary(MasDatosNulim) #podemos tener un resumen estadï¿½stico de las variables del dataset y observamos el cambio en la columna clasificaciï¿½n

mejores_modelos  = regsubsets(classification~.,MasDatosNulim ,nvmax=24) # asigamos la variable dependiente , el dataset y el numero de caracteristicas para encontrar el  mejor modelo

summary(mejores_modelos) # hacemos un resumen estadistico para observar los 25 modelos creados

names(summary(mejores_modelos)) # metricas que se pueden tomar en cuenta para eligir el mejor modelo

summary(mejores_modelos)$adjr2 # selecciona los mejores modelos de acuerdo al valor de r^2 ajustado

which.max(summary(mejores_modelos)$adjr2) # selecciona el modelo con el valor mas grande de r^2 ajustado (el mejor modelo)

summary(mejores_modelos)$adjr2[12] # valor de r^2 ajustado del modelo 12

#observar de forma grafica los 24 modelos creados y cual de ellos se seï¿½ala como el mejor

p <- ggplot(data = data.frame(n_predictores = 1:24,
                              R_ajustado = summary(mejores_modelos)$adjr2),
            aes(x = n_predictores, y = R_ajustado)) +
  geom_line() +
  geom_point()


#Se identifica en rojo el mï¿½ximo
p <- p + geom_point(aes(
  x = n_predictores[which.max(summary(mejores_modelos)$adjr2)],
  y = R_ajustado[which.max(summary(mejores_modelos)$adjr2)]),
  colour = "red", size = 3)
p <- p +  scale_x_continuous(breaks = c(1:24)) + 
  theme_bw() +
  labs(title = 'R2_ajustado vs nï¿½mero de predictores', 
       x =  'nï¿½mero predictores')
p


summary(mejores_modelos)$adjr2[6] # valor de r^2 ajustado del modelo 6

summary(mejores_modelos)$adjr2[7] # valor de r^2 ajustado del modelo 7

summary(mejores_modelos)$adjr2[8] # valor de r^2 ajustado del modelo 8

# al realizar este proceso podemos comprobar que usar un modelo de 12 o 8 caracteristicas 
# arrojan en mismo valor de r^2 ajustado
# ya que Acorde al principio de parsimonia, el modelo que se debe seleccionar 
# como adecuado es el que contiene entre 6-8 predictores

OPTIMO <- read_excel("C:/Users/Sergio/Desktop/ochocaract.xlsx") # cargamos el dataset con 8 variables predictores

colnames(OPTIMO)[9] <- "classification"  #linea que permite acceder a la columna clasifiaciï¿½n

OPTIMO$classification <- factor(OPTIMO$classification, levels = c("0", "1"), labels = c("notckd", "ckd")) # convertir la variable respuesta en factor

OPTIMO[, c(1:8)] <- scale(MasDatosNulim[, c(1:8)])

str(OPTIMO) #comprabamos que la columna classification se convirtio en factor


table(OPTIMO$classification) #nos muestra dentro de nuestros datos que hay sanos y enfermos

#representacion grafica de los datos disponibles

# sg vs al 
ggplot(data = OPTIMO) +
  geom_point(mapping = aes(x = sg, y = al, color = classification))

# rbc vs bu 
ggplot(data = OPTIMO) +
  geom_point(mapping = aes(x = rbc, y = bu, color = classification))

# hemo vs rc 
ggplot(data = OPTIMO) +
  geom_point(mapping = aes(x = hemo, y = rc, color = classification))


# htn vs dm 
ggplot(data = OPTIMO) +
  geom_point(mapping = aes(x = htn, y = dm, color = classification))


#podemos concluir que estamos en un caso de datos que tienen tendencia a ser linealmente no separables

#visualizacion Grafica de la cantidad de pacientes que padecen y no la enfermedad

ggplot(data = OPTIMO, aes(x = classification, y = ..count.., fill = classification)) +
  geom_bar() +
  scale_fill_manual(values = c("gray50", "orangered2")) +
  labs(title = "clasificacion") +
  theme_bw() +
  theme(legend.position = "bottom")


split <- sample.split(OPTIMO$classification, SplitRatio = 0.70) #asignar el 70% de los datos para entrenamiento


training_set <- subset(OPTIMO, split == TRUE) #creacion de los datos asignados para el entrenamiento

test_set <- subset(OPTIMO, split == FALSE) #creacion de los datos asignados para las pruebas

table(training_set$classification) # ver cuantos valores asigno para el entrenamiento

table(test_set$classification) # ver cuantos valores asigno para las pruebas


# verificar que la distribuciï¿½n de la variable respuesta es similar en el conjunto de entrenamiento y prueba

prop.table(table(training_set$classification))

prop.table(table(test_set$classification))

# seleccion de parametros kernel : linear 

tuning <- tune(svm, classification ~ ., data = training_set, 
               kernel = "linear", 
               ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 50)), 
               scale = TRUE)


modelo_svmlineal <- tuning$best.model # nos muestra el mejor parametro
summary(modelo_svmlineal)

#creacion y entrenamiento del modelo

modelo.svc <- svm(classification ~ ., data = training_set, 
                  kernel = "linear", 
                  cost = 1, 
                  scale = TRUE)
modelo.svc

# al tratarse de un problema con más de dos predictores, podemos representar el modelo usando la función plot(), pero creando representaciones 
#entre pares de predictores (teniendo en cuenta que plot.svm solo representa 
#predictores continuos). 

plot(modelo.svc, test_set, sg ~ al)
plot(modelo.svc, test_set, rbc ~ bu)
plot(modelo.svc, test_set, hemo ~ rc)
plot(modelo.svc, test_set, htn ~ dm)

#prediccion sobre el entrenamiento

pred <- predict(modelo.svc, newdata = training_set) 
#creacion de las predicciones del modelo creado  con los datos asignados para entrenamiento

pred

paste("Error de entrenamiento:", 
      100 * mean(training_set$classification != pred) %>% round(digits = 2), "%")

confusionMatrix( data = pred, reference = training_set$classification, positive = "ckd") #visualizacion de la Matriz de confusion tecnica que permite medir la precision del Modelo


# prediccion sobre las pruebas


pred_test <- predict(modelo.svc, newdata = test_set) #creacion de las predicciones del modelo creado  con los datos asignados para pruebas


pred_test
paste("Error de entrenamiento:", 
      100 * mean(test_set$classification != pred_test) %>% round(digits = 2), "%")

confusionMatrix( data = pred_test, reference = test_set$classification, positive = "ckd") #visualizacion de la Matriz de confusion tecnica que permite medir la precision del Modelo

# Error de test
error_test <- mean(predicciones_raw != test_set$classification)




