#*****Importar Librerías*****
#paquete NaiveBayes
library(e1071)
#paquete para división de datos, preprocesamiento, selección de características, ajuste de modelo 
library(caret) 
#paquete para crear modelos de conjunto de listas de modelos de caret
library(caretEnsemble)
#Paquete para leer archivo
library(readxl)
#paquete para análisis de datos
library(tidyverse)
#paquete para crear gráficos
library(ggplot2)
# paquete para el análisis multivariado y la construcción de escalas usando análisis factorial, 
# análisis de componentes principales, análisis de conglomerados y análisis de confiabilidad.
library(psych)
#paquete para matriz de trazado por pares,  matriz de trazado por pares de dos grupos, 
#diagrama de coordenadas paralelas, diagrama de supervivencia y varias funciones para trazar redes.
library(GGally)
#paquete para visualización gráfica de una matriz de correlación
library(corrplot)

#*****Cargar el datset y visualizarlo*****
MasDatosNu <- read_excel("Tesis/Kidney Disease Dataset/EXCEL/MasDatosNu.xlsx")
View(MasDatosNu)

#*****Matriz de Correlación*****
MasDatosNu.cor <- cor(MasDatosNu, method = "spearman")
round(MasDatosNu.cor, digits = 2)
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD","#4477AA"))
corrplot(MasDatosNu.cor, tl.srt = 45, col = col(200), order = "AOE")

#*****Convertir variable Dependiente*****
colnames(MasDatosNu)[26] <- "classification"
MasDatosNu$classification <- factor(MasDatosNu$classification, levels = c("1", "0"), labels = c("CKD", "NotCKD"))

#*****Resumen estadístico*****
summary(MasDatosNu)

#***** Análisis exploratorio de los datos*****
#Visual 1
ggplot(MasDatosNu, aes(age, colour = classification)) + geom_freqpoly(binwidth = 1) + labs(title="Age Distribution by classification")
#visual 2
ggpairs(MasDatosNu)

#*****Escalar variables*****
MasDatosNu[, c(1:25)] <- scale(MasDatosNu[, c(1:25)])
summary(MasDatosNu)

#*****Establecer semilla*****
set.seed(12345)

#*****Partición de la data*****
trainclass <- createDataPartition(MasDatosNu$classification, p=0.7 , list=F) 
training <- MasDatosNu[trainclass,]
testing <- MasDatosNu[-trainclass,]

#*****Verificar dimensiones de la división*****
prop.table(table(MasDatosNu$classification)) * 100

prop.table(table(training$classification)) * 100

prop.table(table(testing$classification)) * 100

#*****Creando el modelo*****
NBModel= naiveBayes(classification ~ ., data = MasDatosNu[trainclass,])
#Visualizar
NBModel

#*****Validación y Predicción*****
pred <- predict(NBModel, MasDatosNu[-trainclass,])
pred

#*****Evaluación del Modelo*****
matriz <-table(MasDatosNu[-trainclass,]$classification, pred, dnn = c("Actual", "Predicha"))
confusionMatrix(matriz)
plot(matriz)

#*****Error de test*****
error <- mean(pred != MasDatosNu[-trainclass,]$classification)
paste("Error de Test del Modelo:", round(error*100, 2), "%")
