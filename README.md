# Play_Katago_with_DeepLearning

This repository contains my **personal experiments and model versions** for the **Deep Learning project** in the Master 2 *Intelligence Artificielle, Systèmes et Données* at **Université Paris Dauphine – PSL**.

The goal of the project is to **train a neural network to play the game of Go** by predicting:
- **Policy**: the probability distribution over all possible moves (19x19 = 361 positions).
- **Value**: the estimated probability for White to win.

All experiments are constrained by the following rules:
- Input and output dimensions must remain unchanged.  
- The total number of trainable parameters must be **below 100,000**.  
- The model must output **two heads**: `policy` and `value`.  
- Evaluation metrics are **categorical accuracy** (for policy) and **mean absolute error (MAE)** (for value).

---

## 🧠 Project Description (from course materials)

This is an academic project from the **Deep Learning class** taught by **Professor Tristan Cazenave**.  
The original project description and resources are available here:  
👉 [Project link](https://www.lamsade.dauphine.fr/~cazenave/DeepLearningProject.html)

> **Course and Project Context:**
>
> - The goal is to train a network for playing the game of Go.  
> - In order to ensure fairness in training resources, networks must have **fewer than 100,000 parameters**.  
> - The maximum number of students per team is two.  
> - The data used for training comes from the **KataGo** self-play dataset (1,000,000 games).  
> - The **input data** is composed of **31 planes of size 19x19**:
>   - color to play  
>   - ladders  
>   - current state on two planes  
>   - two previous states on multiple planes  
> - The **output targets** are:
>   - **Policy:** a vector of size 361 (1.0 for the move played, 0.0 otherwise)
>   - **Value:** a float between 0.0 and 1.0 representing the probability for White to win  
> - The project runs on **Ubuntu 22.04** using **TensorFlow 2.9** and **Keras**.  
> - An example convolutional network with two heads is provided in `golois.py`.  
> - The designed networks must also have the same two-head structure and be saved in `.h5` format.  
> - Optional: To compile the `golois` library, install **Pybind11** and run `compile.sh`.  
> - **Tournaments:**  
>   - Organized every two weeks using the submitted models.  
>   - Models are used by a PUCT engine with 2 seconds of CPU time per move.  
>   - Round-robin tournaments determine the performance of each model.

---

## ⚙️ Technical Setup

- **Operating system:** Ubuntu 22.04  
- **Frameworks:** TensorFlow 2.9, Keras  
- **Languages:** Python 3  
- **Training:** CPU/GPU supported  
- **Input shape:** `(19, 19, 31)`  
- **Outputs:**
  - `policy` → vector of shape `(361,)`
  - `value` → scalar in `[0, 1]`
