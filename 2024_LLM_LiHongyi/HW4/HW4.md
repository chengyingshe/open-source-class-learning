## Prompt

### Zero-shot

#### Prompt 1

```
Task:
Solve the following math problem step-by-step and provide a clear explanation for each step.

Problem: 
{{question}}

Answer:

```

```
Trial 1, accurate_count: 18, total_count: 30, accuracy: 60.0%
Trial 2, accurate_count: 17, total_count: 30, accuracy: 56.666666666666664%
Trial 3, accurate_count: 17, total_count: 30, accuracy: 56.666666666666664%
Final Accuracy: 63.33333333333333%
```

#### Prompt 2

```
Task:
You are a mathematics expert. Your task is to solve the given math problem step-by-step and provide a clear explanation for each step. Ensure that your solution is detailed, logical, and easy to follow.

Problem: 
{{question}}

Answer:
```

```
Trial 1, accurate_count: 21, total_count: 30, accuracy: 70.0%
Trial 2, accurate_count: 19, total_count: 30, accuracy: 63.33333333333333%
Trial 3, accurate_count: 19, total_count: 30, accuracy: 63.33333333333333%
Final Accuracy: 66.66666666666666%
```

### Prompt 3

```
Task:
You are a mathematics expert. Your task is to solve the given math problem step-by-step and provide a clear explanation for each step. Ensure that your solution is detailed, logical, and easy to follow. You need to repeat the solution several times and compare, then provide the final result of the problem after the solutions.

Problem: 
{{question}}

Answer:

```

```
Trial 1, accurate_count: 22, total_count: 30, accuracy: 73.33333333333333%
Trial 2, accurate_count: 21, total_count: 30, accuracy: 70.0%
Trial 3, accurate_count: 23, total_count: 30, accuracy: 76.66666666666667%
Final Accuracy: 76.66666666666667%
```

### In-Context Learning

```
Q:Shawn has five toys.For Christmas,he got two toys each from his mom
and dad.How many toys doe she have now?
A:He has 5 toys.He got 2 from mom,so after that he has 5+2=7 toys.Then
he got 2 more from dad,so in total he has 7+2=9 toys.The answer is 9.
Q:{{question}}
A:
```

```
Trial 1, accurate_count: 12, total_count: 30, accuracy: 40.0%
Trial 2, accurate_count: 14, total_count: 30, accuracy: 46.666666666666664%
Trial 3, accurate_count: 14, total_count: 30, accuracy: 46.666666666666664%
Final Accuracy: 53.333333333333336%
```

### Emotional Stimuli (情绪勒索)

```
Q:{{question}}

This is very important to my career.

A:
```

```
Trial 1, accurate_count: 10, total_count: 30, accuracy: 33.33333333333333%
Trial 2, accurate_count: 13, total_count: 30, accuracy: 43.333333333333336%
Trial 3, accurate_count: 15, total_count: 30, accuracy: 50.0%
Final Accuracy: 46.666666666666664%
```
