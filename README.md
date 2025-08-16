Agente de IA para CartPole-v1 con Q-Learning

Este repositorio contiene un script en Python que implementa un agente de aprendizaje por refuerzo (RL) para resolver el entorno CartPole-v1 de Gymnasium. El agente utiliza el algoritmo de Q-Learning con discretización del espacio de estados para aprender a balancear un poste sobre un carrito el mayor tiempo posible.



----------------------------------
CONCEPTOS CLAVE
----------------------------------

Aprendizaje por Refuerzo (Reinforcement Learning)
Es un área del Machine Learning donde un agente aprende a tomar acciones en un entorno para maximizar una recompensa acumulada. El agente aprende por prueba y error, sin datos previamente etiquetados.

Q-Learning
Es un algoritmo de RL "off-policy" que busca encontrar la mejor acción a tomar en un estado determinado. Lo hace aprendiendo una función de calidad de acción, conocida como Q(s, a), que predice la recompensa futura esperada al tomar la acción 'a' en el estado 's'. La tabla que almacena estos valores se conoce como Q-table. La fórmula de actualización es:

Q(s, a) <-- Q(s, a) + alpha * [r + gamma * max_a'(Q(s', a')) - Q(s, a)]

Discretización del Espacio de Estados
El entorno CartPole tiene un espacio de estados continuo (posición del carro, velocidad, ángulo del poste, etc.). Para que un Q-table (que es una estructura de datos finita) pueda manejarlo, el espacio de estados debe ser convertido a un formato discreto. Este script divide cada variable continua en un número finito de "contenedores" o "bins" (BINS), transformando los valores flotantes en un tupla de enteros que puede usarse como clave en la Q-table.

----------------------------------
FUNCIONAMIENTO DEL SCRIPT
----------------------------------

El código está estructurado en tres fases secuenciales para facilitar la comprensión del proceso de aprendizaje:

Fase 1: Demostración Inicial (Agente sin Entrenar)
- Propósito: Mostrar el comportamiento base del agente.
- Ejecución: Se renderiza el entorno y el agente toma acciones completamente aleatorias. Como es de esperar, el poste se cae casi de inmediato. Se ejecutan 3 intentos para demostrar su consistencia al fallar.

Fase 2: Entrenamiento Rápido y Silencioso
- Propósito: Entrenar al agente de manera eficiente.
- Ejecución: Se inicia un bucle de 20,000 episodios sin renderizado gráfico para acelerar el proceso. En cada paso se actualiza la Q-table utilizando una estrategia épsilon-greedy para balancear la exploración y la explotación.

Fase 3: Demostración Final (Agente Experto)
- Propósito: Evaluar el desempeño del agente ya entrenado.
- Ejecución: Se vuelve a renderizar el entorno. Esta vez, el agente utiliza exclusivamente la Q-table aprendida para tomar la decisión óptima en cada paso. Se podrá observar cómo el agente es capaz de balancear el poste durante mucho más tiempo.

----------------------------------
PARÁMETROS CLAVE (HIPERPARÁMETROS)
----------------------------------

- EPISODIOS: 20,000 - El número total de episodios para el entrenamiento.
- ALPHA: 0.1 - La tasa de aprendizaje (alpha). Controla la rapidez con la que el agente actualiza los valores Q.
- GAMMA: 0.99 - El factor de descuento (gamma). Determina la importancia de las recompensas futuras.
- EPSILON: 1.0 - La probabilidad inicial de exploración.
- EPSILON_DECAY: 0.9999 - El factor por el que EPSILON se multiplica en cada episodio para reducir gradualmente la exploración.
- BINS: [20, 20, 20, 20] - El número de "contenedores" para discretizar cada una de las 4 variables de estado.

----------------------------------
CÓMO EMPEZAR
----------------------------------

Prerrequisitos:
Python instalado con pip:

pip install gymnasium numpy

Ejecución:
Para ejecutar el programa, simplemente corre el script desde tu terminal:

python rl-pole.py

Al ejecutarlo, primero verás una ventana donde el agente falla repetidamente. Luego, la ventana se cerrará y el entrenamiento comenzará en la consola. Finalmente, la ventana volverá a aparecer para mostrar al agente entrenado balanceando el poste con éxito.