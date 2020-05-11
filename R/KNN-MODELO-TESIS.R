library(readxl) #permite leer el dataset en formato excel
library(caret) #permite realizar las divisiones de los datos (entranamiento y prueba) # contiene parametros de la matriz de confusión
library(class) # contiene la funcion KKnn
library(kknn) # funcion para implementar k-vecinos

set.seed(12345) #definimos esta semilla para que cada vez que se ejecute los valores no cambien

                 
                                                              
MasDatosNu <- read_excel("C:/Users/Sergio/Desktop/MasDatosNu.xlsx") #Cargar el dataset en formato excel

str (MasDatosNu) #permite ver el tipo de dato por el que esta conformado del dataset

colnames(MasDatosNu)[26] <- "classification" #linea que permite acceder a la columna clasifiación
MasDatosNu$classification <- factor(MasDatosNu$classification, levels = c("0", "1"), labels = c("notckd", "ckd")) #en la columna clasificación asignar al 0 "notckd" y al 1 "ckd"

str (MasDatosNu) # la columna clasificación se convirtio a factor

MasDatosNu[c(1:294),-1] # eliminamos la columna Id ya que es un dato irrelevante

MasDatosNulim = MasDatosNu[c(1:294),-1] # cargamos un nuevo dataset sin la columna ID

str(MasDatosNulim) # observamos que la columna ID fue eliminada 

summary(MasDatosNulim) # resumen estadistico de las variables del dataset

#se observa  que existe cierta variabilidad y diferencia de escala entre los valores
#mínimos y máximos de los registros medidos, por lo que es conveniente, 
# el rescalar las variables numéricas a fin de que ninguna tenga 
#mayor influencia sobre otra como predictor al momento de construir los modelos de clasificación. 
#Entonces, usemos la función scale:

MasDatosNulim[, c(1:24)] <- scale(MasDatosNulim[, c(1:24)])
summary(MasDatosNulim) #podemos observar que los valores ya no varian tanto

# Se crean los índices de las observaciones de entrenamiento
train <- createDataPartition(y = MasDatosNulim$classification, p = 0.70, list = FALSE, times = 1)
datos_train <- MasDatosNulim[train, ] # 70% para el entrenamiento 
datos_test  <- MasDatosNulim[-train, ] # 30% para las pruebas 

modelo <- train.kknn(classification ~ ., data = datos_train, kmax = 13) # diseño del Modelo con un valor maximo de 13

modelo # detalles del modelo donde selecciona el mejor valor de k para los datos analizados

plot(modelo) # en el grafico se muestra el valor mas optimo de k

#pruebas con el valor de establecido de k=3

# Aplicamos el algoritmo K-NN seleccionando 3 como k inicial

k3 <- knn(datos_train[,1:24],datos_test[,1:24], datos_train$classification , k = 3, prob = TRUE )

#prediccion con los valores de entrenamiento 

pred <- predict(modelo, datos_train[, -25]) # predicción con los datos de Prueba
pred

# evaluacion del entrenamiento 

confusionMatrix( data = pred, reference = datos_train$classification, positive = "ckd") #visualizacion de la Matriz de confusion tecnica que permite medir la precision del Modelo

#prediccion con los valores de pruebas 

pred_t <- predict(modelo, datos_test[, -25]) # predicción con los datos de Prueba
pred

# evalucion de las pruebas 

confusionMatrix( data = pred_t, reference = datos_test$classification, positive = "ckd") #visualizacion de la Matriz de confusion tecnica que permite medir la precision del Modelo






