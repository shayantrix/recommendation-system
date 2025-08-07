# Book Recommendation System

Welcome! This project is a **Book Recommendation System**. It can suggest books to you based on what you like, or show you books similar to another book. You can use it with a simple web page!

---

## How to Use This Project

### 1. **Install Python**

First, make sure you have **Python 3.8 or newer** installed on your computer.  
If you don’t, download it from [python.org](https://www.python.org/downloads/).

---

### 2. **Download the Project Files**

- Download all the files and folders for this project.
- Make sure you have the `data/BookRating` folder with the CSV files (`Books.csv`, `Users.csv`, `Ratings.csv`).
- You should also have the `recommendation_model.pkl` file (the trained model).

---

### 3. **Install the Needed Libraries**

Open a terminal (Command Prompt) in the project folder and run:

```
pip install -r requirements.txt
```

This will install everything you need!

---

### 4. **Start the Recommendation App**

In the terminal, run:

```
python src/app.py
```

Wait a few seconds. You will see a message like:

```
Running on local URL:  http://127.0.0.1:7860
```

---

### 5. **Open the App in Your Browser**

- Open your web browser.
- Go to the address shown in the terminal (usually [http://127.0.0.1:7860](http://127.0.0.1:7860)).
- You will see a simple web page where you can enter a **book title** or **user ID** to get recommendations!

---

## What Each File Does

- **src/app.py**: The main program. It loads the model and data, and starts the web app.
- **data/BookRating/Books.csv, Users.csv, Ratings.csv**: The data about books, users, and ratings.
- **recommendation_model.pkl**: The trained model that knows how to recommend books.
- **requirements.txt**: The list of libraries you need to install.

---

## How the Code Works (Simple Explanation)

1. **Loads the data** about books, users, and ratings.
2. **Loads the trained model** that learned from the data.
3. **Starts a web app** using Gradio, so you can use the recommender in your browser.
4. When you enter a **book title**, it finds similar books.

---

## requirements.txt

Put this in a file called `requirements.txt` in your project folder:

```
fastai==2.7.13
torch==2.2.2
pandas==2.2.2
numpy==1.26.4
gradio==4.29.0
matplotlib==3.8.4
scikit-learn==1.4.2
```

---

## Tips

- If you get an error about a missing library, run `pip install <library-name>` in your terminal.
- If you want to stop the app, press `Ctrl+C` in the terminal.

---

## That’s it!

This program is written by Shayan Amir Shahkarami, using fastai, an amazing library for deep learning.
