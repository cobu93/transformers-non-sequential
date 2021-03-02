Metodología para evaluación de transformadores en información no secuencial:


1. Desarrollo de baselines utilizando multiples modelos (redes neuronales, clusterización, árboles de decisión, etc.) para un problema seleccionado. La idea de hacer múltiples baselines es tratar de mejorar el modelo utilizando un transformador lo cuál podría tener un mayor impacto en los resultados. 

2. Implementar el aprendizaje utilizando un transformador. Se trata de mantaner el transformador lo más simple (en cuánto a cantidad de heads, dimensiones de embeddings y neuronas en redes feedforward) posible para facilitar la interpretación de la matriz de atención. 

3. Se extraen las matrices de atención para todo el conjunto de entrenamiento. En caso de utilizar la implementación de PyTorch esto puede hacerse utilizando:

    **attn_output, attn_output_weights = model.transformer_encoder.layers[0].self_attn(src, src, src)**

    La matriz de atención está dentro de la variable **attn_output_weights**, en la cuál cada fila representa una distribución de probabilidad.

4. El análisis de la atención puede hacerse de manera "libre". Los análisis probados anteriormente son los siguientes:

    - Análizar las matrices de atención agrupándolas por clase predecida. La idea detrás de esta agrupación es que aquellos ejemplos predecidos con la misma clase tienen la atención concentrada en las mismas características.

    - Tratar de hacer la clasificación utilizando todas las matrices de atención como nuevo conjunto de entrenamiento.

    - Analizar las matrices de atención agrupándolas de manera automática con algún método de clusterización.
    

5. Para seleccionar las características de manera automática se utilizó una métrica basada en entropía:

- Dependiendo del número de características se calcula la probabilidad que contendría cada celda de la matriz de atención si siguiera una distribución de probabilidad uniforme:

    Pr(attn_unif(c1)) = Pr(attn_unif(c2)) = ... = Pr(attn_unif(cn)) = 1 / n

    Entropía_uniforme = -log(Pr(attn_unif(c1)))

- Dado que se asume que la distribución es uniforme, su entropía es la más alta que puede existir. Se calcula la entropía de la distribución uniforme de acuerdo al número de características.

- Para la matriz de atención se calcula la entropía de cada fila y se compara con la entropía de la distribución uniforme. Si se cumple la siguiente condición para un a entre [0, 1] asignado para el problema (es decir, la entropía de la fila es menor a la entropía uniforme y por lo tanto la probabilidad está cargada en alguna(s) característica(s) especifica(s)) se considera la fila como "relevante".

    Entropía_predecida = Pr(attn(c1)) * log(Pr(attn(c1))) + Pr(attn(c2)) * log(Pr(attn(c2))) + ...

a * Entropía_uniforme < Entropía_predecida


- Para las filas relevantes, se comparan todas las probabilidades con la probabilidad en la distribución uniforme, si para b entre (1, infinito) (seleccionado por el usuario) se cumple la siguiente condición considera una característica importante y se calculan las celdas importantes.

    Pr(attn(cx)) > b * Pr(attn_unif(c1)) con x entre 1 y el número de características.

- Una vez calculadas las filas y las columnas relevantes, se seleccionan las características con mayor atención y se genera la máscara.

6. Hasta ahora con las máscaras se ha realizado la selección de características. Para algunos modelos, con esta selección de características sobre el dataset original, se ha observado un mejor desempeño que en la baseline (utilizando todas las características).
