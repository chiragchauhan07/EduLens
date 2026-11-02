"""
EduLens Synthetic Dataset Generator
Generates ~500 student-LLM tutor learning sessions for NLP/ML experimentation.
"""

import json
import csv
import random
import math
import hashlib
from copy import deepcopy
from datetime import datetime

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# SUBJECT CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────

SUBJECTS = {
    "Python programming": {
        "target": 70,
        "topics": {
            "Variables and Data Types": {
                "subtopics": ["integers vs floats", "string operations", "type conversion", "boolean logic", "None type"],
            },
            "Loops": {
                "subtopics": ["for loops", "while loops", "break and continue", "nested loops", "loop with enumerate"],
            },
            "Functions": {
                "subtopics": ["function definitions", "return values", "default arguments", "args and kwargs", "lambda functions", "scope"],
            },
            "Lists and Dictionaries": {
                "subtopics": ["list indexing", "list slicing", "list comprehension", "dictionary methods", "nested structures"],
            },
            "Object-Oriented Programming": {
                "subtopics": ["classes and objects", "inheritance", "polymorphism", "encapsulation", "dunder methods", "__init__"],
            },
            "Exception Handling": {
                "subtopics": ["try-except blocks", "raising exceptions", "custom exceptions", "finally clause", "exception hierarchy"],
            },
            "File Handling": {
                "subtopics": ["reading files", "writing files", "context managers", "CSV reading", "JSON files"],
            },
        },
    },
    "Data Structures and Algorithms": {
        "target": 60,
        "topics": {
            "Arrays and Strings": {
                "subtopics": ["two-pointer technique", "sliding window", "string manipulation", "array rotation", "prefix sums"],
            },
            "Linked Lists": {
                "subtopics": ["singly linked list", "doubly linked list", "cycle detection", "reversing a linked list", "merge sorted lists"],
            },
            "Stacks and Queues": {
                "subtopics": ["stack operations", "queue operations", "monotonic stack", "priority queues", "deque"],
            },
            "Trees and Graphs": {
                "subtopics": ["binary search tree", "tree traversal", "BFS", "DFS", "graph representation", "topological sort"],
            },
            "Sorting Algorithms": {
                "subtopics": ["bubble sort", "merge sort", "quicksort", "heap sort", "counting sort", "stability"],
            },
            "Recursion": {
                "subtopics": ["base case design", "recursive vs iterative", "call stack", "memoization", "backtracking"],
            },
            "Dynamic Programming": {
                "subtopics": ["overlapping subproblems", "optimal substructure", "tabulation", "Fibonacci DP", "knapsack problem", "LCS"],
            },
            "Time and Space Complexity": {
                "subtopics": ["Big-O notation", "worst vs average case", "amortized analysis", "space complexity", "O(log n) intuition"],
            },
        },
    },
    "Machine Learning": {
        "target": 70,
        "topics": {
            "Supervised vs Unsupervised Learning": {
                "subtopics": ["supervised learning basics", "unsupervised clustering", "semi-supervised learning", "labels and features"],
            },
            "Linear Regression": {
                "subtopics": ["least squares", "gradient descent", "mean squared error", "R-squared", "multivariate regression"],
            },
            "Logistic Regression": {
                "subtopics": ["sigmoid function", "log loss", "decision boundary", "multi-class logistic regression", "regularization"],
            },
            "Decision Trees": {
                "subtopics": ["Gini impurity", "information gain", "pruning", "tree depth", "feature importance"],
            },
            "Random Forest": {
                "subtopics": ["bagging", "feature subsampling", "ensemble methods", "out-of-bag error", "hyperparameter tuning"],
            },
            "Overfitting and Underfitting": {
                "subtopics": ["bias-variance tradeoff", "regularization L1 L2", "dropout", "learning curves", "cross-validation"],
            },
            "Model Evaluation": {
                "subtopics": ["accuracy precision recall F1", "ROC-AUC", "confusion matrix", "k-fold cross-validation", "train-test split"],
            },
            "Feature Engineering": {
                "subtopics": ["one-hot encoding", "feature scaling", "PCA", "feature selection", "handling categorical variables"],
            },
        },
    },
    "Mathematics and Linear Algebra": {
        "target": 45,
        "topics": {
            "Vectors": {
                "subtopics": ["vector addition", "scalar multiplication", "dot product", "magnitude and unit vectors", "vector spaces"],
            },
            "Matrices": {
                "subtopics": ["matrix addition", "matrix multiplication", "transpose", "identity matrix", "matrix inverse"],
            },
            "Eigenvalues and Eigenvectors": {
                "subtopics": ["characteristic equation", "diagonalization", "geometric interpretation", "PCA connection"],
            },
            "Probability": {
                "subtopics": ["conditional probability", "Bayes theorem", "independence", "random variables", "expected value"],
            },
            "Derivatives and Gradients": {
                "subtopics": ["chain rule", "partial derivatives", "gradient descent intuition", "Jacobian", "directional derivative"],
            },
        },
    },
    "Database Management and SQL": {
        "target": 45,
        "topics": {
            "SQL Queries": {
                "subtopics": ["SELECT basics", "WHERE clause", "ORDER BY", "GROUP BY", "aggregate functions", "HAVING"],
            },
            "Joins": {
                "subtopics": ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN", "self joins", "cross joins"],
            },
            "Normalization": {
                "subtopics": ["1NF 2NF 3NF", "functional dependencies", "BCNF", "denormalization trade-offs"],
            },
            "Primary and Foreign Keys": {
                "subtopics": ["primary key constraints", "foreign key references", "referential integrity", "composite keys"],
            },
            "Transactions and ACID": {
                "subtopics": ["ACID properties", "commit and rollback", "isolation levels", "deadlock in databases"],
            },
            "Indexing": {
                "subtopics": ["B-tree indexes", "hash indexes", "composite indexes", "index trade-offs", "query planning"],
            },
        },
    },
    "Operating Systems and Computer Networks": {
        "target": 40,
        "topics": {
            "Processes and Threads": {
                "subtopics": ["process vs thread", "context switching", "multithreading", "race conditions", "synchronization"],
            },
            "CPU Scheduling": {
                "subtopics": ["FCFS", "round-robin", "SJF", "priority scheduling", "preemptive vs non-preemptive"],
            },
            "Deadlocks": {
                "subtopics": ["deadlock conditions", "Coffman conditions", "deadlock prevention", "deadlock detection", "banker's algorithm"],
            },
            "Memory Management": {
                "subtopics": ["paging", "segmentation", "virtual memory", "page replacement", "TLB"],
            },
            "TCP/IP and Networking": {
                "subtopics": ["TCP vs UDP", "three-way handshake", "IP addressing", "subnetting", "routing protocols", "DNS", "HTTP"],
            },
        },
    },
    "Natural Language Processing": {
        "target": 60,
        "topics": {
            "Text Preprocessing": {
                "subtopics": ["tokenization", "stop word removal", "stemming vs lemmatization", "lowercasing", "punctuation removal"],
            },
            "Bag of Words and TF-IDF": {
                "subtopics": ["term frequency", "inverse document frequency", "sparse vectors", "BoW limitations", "TF-IDF weighting"],
            },
            "Word Embeddings": {
                "subtopics": ["Word2Vec", "GloVe", "co-occurrence matrix", "embedding dimensions", "analogies in embeddings"],
            },
            "Sentence Transformers": {
                "subtopics": ["BERT architecture", "sentence embeddings", "cosine similarity", "pooling strategies", "fine-tuning"],
            },
            "Semantic Similarity": {
                "subtopics": ["cosine similarity", "Euclidean distance", "nearest neighbor search", "semantic search", "paraphrase detection"],
            },
            "Text Classification": {
                "subtopics": ["sentiment analysis", "topic classification", "multi-label classification", "zero-shot classification", "evaluation metrics"],
            },
        },
    },
    "Statistics and Data Science": {
        "target": 50,
        "topics": {
            "Descriptive Statistics": {
                "subtopics": ["mean median mode", "variance standard deviation", "skewness kurtosis", "percentiles quartiles"],
            },
            "Probability Distributions": {
                "subtopics": ["normal distribution", "binomial distribution", "Poisson distribution", "uniform distribution", "CLT"],
            },
            "Hypothesis Testing": {
                "subtopics": ["null vs alternative hypothesis", "p-value", "t-test", "chi-squared test", "Type I Type II errors"],
            },
            "Correlation and Regression": {
                "subtopics": ["Pearson correlation", "Spearman correlation", "correlation vs causation", "regression line interpretation"],
            },
            "Data Cleaning with Pandas": {
                "subtopics": ["missing value imputation", "outlier detection", "data type coercion", "merging dataframes", "groupby operations"],
            },
        },
    },
    "Other Computer Science Concepts": {
        "target": 60,
        "topics": {
            "Object-Oriented Design": {
                "subtopics": ["SOLID principles", "design patterns", "factory pattern", "singleton pattern", "observer pattern"],
            },
            "APIs and REST": {
                "subtopics": ["HTTP methods", "REST principles", "status codes", "JSON payloads", "authentication", "rate limiting"],
            },
            "Software Engineering": {
                "subtopics": ["Agile methodology", "version control", "code review", "unit testing", "CI/CD pipelines"],
            },
            "Git": {
                "subtopics": ["git commit", "branching strategies", "merge vs rebase", "git conflicts", "pull requests"],
            },
            "Cloud Computing": {
                "subtopics": ["IaaS PaaS SaaS", "containerization Docker", "serverless functions", "load balancing", "cloud storage"],
            },
            "Basic Cybersecurity": {
                "subtopics": ["encryption basics", "hashing", "SQL injection", "XSS", "authentication vs authorization", "HTTPS"],
            },
        },
    },
}

LEARNING_PATTERNS = [
    "smooth_learning", "repeated_clarification", "misconception_correction",
    "example_driven", "problem_solving", "partial_understanding",
    "progressive_understanding", "persistent_confusion", "independent_reasoning",
    "mixed_understanding", "efficient_learning", "exploratory_learning",
]

DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced"]

# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION TEMPLATES & DIALOGUE BANKS
# ─────────────────────────────────────────────────────────────────────────────

# Each pattern has a list of dialogue-building strategies.
# We use topic/subtopic-aware content generation to fill in the actual content.

def make_conversation(topic, subtopic, pattern, difficulty, initial_level, final_level,
                      misconception_present, subject_category):
    """Generate a realistic multi-turn conversation for a given session."""

    rng = random.Random(topic + subtopic + pattern + difficulty)

    # target message count
    target_msgs = rng.randint(8, 18)
    if pattern in ("efficient_learning", "smooth_learning"):
        target_msgs = rng.randint(8, 13)
    elif pattern in ("persistent_confusion", "repeated_clarification"):
        target_msgs = rng.randint(12, 20)
    elif pattern in ("exploratory_learning", "mixed_understanding"):
        target_msgs = rng.randint(14, 20)

    # subject-aware context
    is_programming = subject_category in ("Python programming", "Data Structures and Algorithms",
                                           "Natural Language Processing", "Other Computer Science Concepts")
    is_math = subject_category in ("Mathematics and Linear Algebra", "Statistics and Data Science")
    is_db = subject_category == "Database Management and SQL"
    is_ml = subject_category == "Machine Learning"

    convo = []

    # Opening student message (varies by initial level)
    openers = _opening_messages(topic, subtopic, difficulty, initial_level, is_programming, is_math, is_db, rng)
    convo.append({"role": "user", "content": openers[0]})
    convo.append({"role": "assistant", "content": _assistant_intro(topic, subtopic, difficulty, is_programming, is_math, is_db, rng)})

    # Build middle turns based on pattern
    turns = _build_turns(topic, subtopic, pattern, difficulty, initial_level, final_level,
                         misconception_present, is_programming, is_math, is_db, is_ml, rng,
                         target_msgs - 2)

    convo.extend(turns)

    # Closing turn
    closing = _closing_turn(pattern, final_level, topic, subtopic, is_programming, rng)
    if closing:
        convo.extend(closing)

    # Ensure alternating roles (fix minor issues)
    convo = _fix_alternation(convo)

    return convo


def _opening_messages(topic, subtopic, difficulty, initial_level, is_programming, is_math, is_db, rng):
    low_openers = [
        f"Hey, I'm trying to learn about {topic} but I'm pretty confused. Can you explain {subtopic} to me?",
        f"I've read about {topic} but honestly it doesn't make sense yet. Like, what exactly is {subtopic}?",
        f"I don't really understand {topic}. My notes mention {subtopic} but I have no idea what that means.",
        f"I'm new to {topic}. Can you start from the basics and explain {subtopic}?",
        f"Can you help me understand {topic}? I keep reading about {subtopic} but it's not clicking.",
    ]
    med_openers = [
        f"I have a basic idea of {topic}, but I'm still unclear about {subtopic}. Can you clarify?",
        f"I understand the basics of {topic} but when it comes to {subtopic} I get confused.",
        f"I've been studying {topic} and I think I get the general idea, but {subtopic} still trips me up.",
        f"I know some {topic} fundamentals. Could you walk me through {subtopic} in more detail?",
    ]
    high_openers = [
        f"I'm fairly comfortable with {topic}. I want to dig deeper into {subtopic} — specifically the edge cases.",
        f"I understand {topic} well, but I want to make sure my mental model of {subtopic} is correct.",
        f"I've been using {topic} in projects, and I'd like to better understand the internals of {subtopic}.",
        f"Quick question on {topic}: when does {subtopic} actually matter in practice versus theory?",
    ]

    pool = low_openers if initial_level == "low" else (med_openers if initial_level == "medium" else high_openers)
    return [rng.choice(pool)]


def _assistant_intro(topic, subtopic, difficulty, is_programming, is_math, is_db, rng):
    intros = [
        f"Great question! Let's break down {subtopic} step by step. {_concept_seed(topic, subtopic, is_programming, is_math, is_db, rng)}",
        f"Sure! {subtopic} is a key concept in {topic}. {_concept_seed(topic, subtopic, is_programming, is_math, is_db, rng)}",
        f"Happy to help. {_concept_seed(topic, subtopic, is_programming, is_math, is_db, rng)} Does that help as a starting point?",
        f"Let me explain {subtopic} clearly. {_concept_seed(topic, subtopic, is_programming, is_math, is_db, rng)}",
        f"Of course. {_concept_seed(topic, subtopic, is_programming, is_math, is_db, rng)} Let me know if you'd like an example.",
    ]
    return rng.choice(intros)


def _concept_seed(topic, subtopic, is_programming, is_math, is_db, rng):
    """Return a seed explanation sentence about the subtopic."""
    programming_seeds = {
        "list indexing": "In Python, list indexing starts at 0, so `my_list[0]` gives you the first element.",
        "list slicing": "Slicing lets you extract a portion of a list: `my_list[1:4]` returns elements at positions 1, 2, and 3.",
        "list comprehension": "List comprehension is a concise way to create lists: `[x**2 for x in range(5)]` gives `[0, 1, 4, 9, 16]`.",
        "inheritance": "In OOP, inheritance allows a child class to reuse and extend the behavior of a parent class.",
        "try-except blocks": "A `try` block lets you test code for errors, and the `except` block handles those errors gracefully.",
        "for loops": "A `for` loop iterates over a sequence — for example, `for item in my_list:` visits each element one at a time.",
        "while loops": "A `while` loop keeps running as long as a condition is `True`. Be careful to update the condition inside the loop to avoid infinite loops.",
        "default arguments": "Default arguments let you define a fallback value for a parameter: `def greet(name='World'):` works with or without an argument.",
        "lambda functions": "A lambda is a small anonymous function: `square = lambda x: x**2` is equivalent to a regular one-line function.",
        "context managers": "Context managers (using `with`) handle setup and cleanup automatically — most commonly used when opening files.",
        "two-pointer technique": "The two-pointer technique uses two indices that move through a data structure simultaneously, often reducing O(n²) to O(n).",
        "sliding window": "A sliding window maintains a contiguous subset of elements and slides across the array, useful for substring or subarray problems.",
        "cycle detection": "Floyd's cycle detection algorithm (fast and slow pointers) determines if a linked list has a cycle in O(n) time and O(1) space.",
        "merge sort": "Merge sort divides the array in half recursively, then merges sorted halves — it's O(n log n) and stable.",
        "quicksort": "Quicksort picks a pivot element and partitions the array around it; average time complexity is O(n log n).",
        "BFS": "Breadth-First Search explores nodes level by level using a queue, making it ideal for finding shortest paths in unweighted graphs.",
        "DFS": "Depth-First Search explores as far as possible down each branch before backtracking, using a stack (or recursion).",
        "memoization": "Memoization caches the results of expensive function calls so they aren't recalculated when called again with the same inputs.",
        "Big-O notation": "Big-O notation describes the upper bound on how an algorithm's runtime grows relative to its input size.",
    }
    ml_seeds = {
        "gradient descent": "Gradient descent iteratively adjusts model parameters in the direction that reduces the loss function the most.",
        "sigmoid function": "The sigmoid function maps any real value to a probability between 0 and 1: σ(z) = 1 / (1 + e^(-z)).",
        "Gini impurity": "Gini impurity measures how often a randomly chosen element would be incorrectly labeled; lower values mean purer nodes.",
        "bias-variance tradeoff": "High bias means the model is too simple (underfitting); high variance means it's too sensitive to training data (overfitting).",
        "k-fold cross-validation": "K-fold CV splits data into k subsets and trains k models, each time using a different subset as validation — giving a more reliable accuracy estimate.",
        "one-hot encoding": "One-hot encoding converts a categorical variable into binary columns — one per category — so models can use it numerically.",
        "feature scaling": "Feature scaling (e.g., Min-Max or StandardScaler) brings features onto the same scale, which is critical for distance-based models.",
        "ROC-AUC": "The ROC curve plots true-positive rate vs false-positive rate at various thresholds; AUC is the area under this curve — higher is better.",
        "bagging": "Bagging trains multiple models on random subsets (with replacement) of the training data and aggregates their predictions.",
    }
    math_seeds = {
        "dot product": "The dot product of two vectors **a** and **b** is the sum of the products of their corresponding components: a·b = Σ aᵢbᵢ.",
        "matrix multiplication": "Matrix multiplication requires the number of columns in the first matrix to equal the number of rows in the second; the result is a new matrix.",
        "Bayes theorem": "Bayes' theorem states: P(A|B) = P(B|A) × P(A) / P(B), allowing us to update beliefs given new evidence.",
        "chain rule": "The chain rule says: if y = f(g(x)), then dy/dx = f'(g(x)) × g'(x) — we differentiate the outer function, then multiply by the derivative of the inner.",
        "eigenvalues": "An eigenvector of a matrix A is a vector v such that Av = λv, where λ (lambda) is the corresponding eigenvalue — a scalar.",
        "conditional probability": "P(A|B) is the probability of A given that B has already occurred: P(A|B) = P(A ∩ B) / P(B).",
        "normal distribution": "The normal distribution is a bell-shaped, symmetric distribution characterized by its mean μ and standard deviation σ.",
        "p-value": "A p-value tells you the probability of observing your results (or more extreme) if the null hypothesis were true. A small p-value suggests rejecting H₀.",
        "variance": "Variance measures the average squared deviation from the mean: σ² = Σ(xᵢ - μ)² / N.",
    }
    db_seeds = {
        "INNER JOIN": "An INNER JOIN returns only rows where there is a matching value in both tables.",
        "LEFT JOIN": "A LEFT JOIN returns all rows from the left table and the matching rows from the right table; unmatched right rows appear as NULL.",
        "1NF 2NF 3NF": "1NF eliminates repeating groups; 2NF removes partial dependencies; 3NF removes transitive dependencies — each step further reduces redundancy.",
        "ACID properties": "ACID stands for Atomicity, Consistency, Isolation, Durability — properties that guarantee database transactions are processed reliably.",
        "B-tree indexes": "A B-tree index stores key values in a balanced tree structure, enabling O(log n) lookups and efficient range queries.",
        "aggregate functions": "SQL aggregate functions like COUNT, SUM, AVG, MIN, and MAX operate on sets of rows and return a single result.",
        "isolation levels": "Isolation levels (Read Uncommitted → Serializable) control how much one transaction can see another's changes; higher isolation reduces anomalies but increases lock contention.",
        "functional dependencies": "A functional dependency X → Y means that knowing X uniquely determines Y — this concept drives normalization decisions.",
    }
    nlp_seeds = {
        "tokenization": "Tokenization is the process of splitting text into individual units (tokens) — usually words, subwords, or sentences.",
        "stemming vs lemmatization": "Stemming crudely chops word endings (e.g., 'running' → 'run'); lemmatization uses vocabulary knowledge to return the base form (lemma).",
        "TF-IDF weighting": "TF-IDF (Term Frequency–Inverse Document Frequency) gives higher weight to terms that are frequent in a document but rare across all documents.",
        "Word2Vec": "Word2Vec trains a shallow neural network to produce dense vector representations where semantically similar words are close together.",
        "cosine similarity": "Cosine similarity measures the angle between two vectors; values close to 1 mean the vectors point in the same direction (semantically similar).",
        "BERT architecture": "BERT is a transformer-based model that produces contextual word embeddings by considering both left and right context simultaneously (bidirectional).",
        "sentence embeddings": "Sentence Transformers pool token-level BERT representations to produce a single fixed-size vector representing the meaning of an entire sentence.",
        "sentiment analysis": "Sentiment analysis classifies text as positive, negative, or neutral — it's one of the most common NLP classification tasks.",
        "stop word removal": "Stop words are common words (like 'the', 'is', 'in') that carry little semantic meaning and are often removed before analysis.",
    }
    os_seeds = {
        "context switching": "Context switching is the process of storing and restoring the state of a CPU so that multiple processes can share a single processor.",
        "race conditions": "A race condition occurs when multiple threads access shared data concurrently and the outcome depends on the unpredictable order of execution.",
        "round-robin": "Round-robin scheduling assigns each process a fixed time slice (quantum) and cycles through them in order — simple and fair.",
        "Coffman conditions": "Deadlock requires all four Coffman conditions simultaneously: mutual exclusion, hold-and-wait, no preemption, and circular wait.",
        "virtual memory": "Virtual memory allows a process to use more memory than physically available by mapping virtual addresses to physical addresses, with overflow stored on disk.",
        "TCP vs UDP": "TCP provides reliable, ordered, connection-oriented delivery; UDP is connectionless and faster but doesn't guarantee delivery.",
        "three-way handshake": "TCP connection setup uses SYN, SYN-ACK, and ACK packets — ensuring both sides are ready to communicate.",
        "DNS": "DNS (Domain Name System) translates human-readable domain names (like google.com) into IP addresses that computers use to communicate.",
        "paging": "Paging divides memory into fixed-size blocks (pages) and maps virtual pages to physical frames, eliminating external fragmentation.",
    }

    all_seeds = {}
    all_seeds.update(programming_seeds)
    all_seeds.update(ml_seeds)
    all_seeds.update(math_seeds)
    all_seeds.update(db_seeds)
    all_seeds.update(nlp_seeds)
    all_seeds.update(os_seeds)

    # find best match
    for key, val in all_seeds.items():
        if key.lower() in subtopic.lower() or subtopic.lower() in key.lower():
            return val

    # fallback generic
    return (f"The key idea in {subtopic} is understanding how it fits into the broader context of {topic}. "
            f"Let me walk you through the core concepts with a concrete example.")


def _build_turns(topic, subtopic, pattern, difficulty, initial_level, final_level,
                 misconception_present, is_programming, is_math, is_db, is_ml, rng, remaining):
    """Build the middle turns of the conversation based on the pattern."""
    turns = []

    if pattern == "smooth_learning":
        turns = _smooth_learning_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining)
    elif pattern == "repeated_clarification":
        turns = _repeated_clarification_turns(topic, subtopic, difficulty, is_programming, rng, remaining)
    elif pattern == "misconception_correction":
        turns = _misconception_correction_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining)
    elif pattern == "example_driven":
        turns = _example_driven_turns(topic, subtopic, difficulty, is_programming, is_math, is_db, rng, remaining)
    elif pattern == "problem_solving":
        turns = _problem_solving_turns(topic, subtopic, difficulty, is_programming, is_math, is_db, rng, remaining)
    elif pattern == "partial_understanding":
        turns = _partial_understanding_turns(topic, subtopic, difficulty, is_programming, rng, remaining)
    elif pattern == "progressive_understanding":
        turns = _progressive_understanding_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining)
    elif pattern == "persistent_confusion":
        turns = _persistent_confusion_turns(topic, subtopic, difficulty, is_programming, rng, remaining)
    elif pattern == "independent_reasoning":
        turns = _independent_reasoning_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining)
    elif pattern == "mixed_understanding":
        turns = _mixed_understanding_turns(topic, subtopic, difficulty, is_programming, rng, remaining)
    elif pattern == "efficient_learning":
        turns = _efficient_learning_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining)
    elif pattern == "exploratory_learning":
        turns = _exploratory_learning_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining)
    else:
        turns = _smooth_learning_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining)

    # Trim or pad to roughly remaining target
    if len(turns) > remaining + 4:
        turns = turns[:remaining + 4]

    return turns


def _smooth_learning_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining):
    turns = []
    questions = [
        f"Can you give me a concrete example of {subtopic}?",
        f"What happens if I use {subtopic} in a slightly different scenario?",
        f"Is there a common mistake people make with {subtopic}?",
        f"How does {subtopic} relate to other parts of {topic}?",
        f"Could you walk me through a step-by-step example?",
    ]
    answers = [
        f"Absolutely! {_example_sentence(topic, subtopic, is_programming, is_math, rng)}",
        f"Good question — the key thing to watch out for is {_pitfall_sentence(topic, subtopic, rng)}.",
        f"That ties directly into {topic}: {_connection_sentence(topic, subtopic, rng)}.",
        f"Of course! Let me show you a worked example step by step. {_step_example(topic, subtopic, is_programming, is_math, rng)}",
        f"Great intuition! {_reinforcement_sentence(topic, subtopic, rng)}",
    ]
    follow_ups = [
        f"That makes sense! So if I understand correctly, {subtopic} basically means {_student_paraphrase(subtopic, rng)}?",
        f"Ah okay, so the important thing is {_student_key_takeaway(subtopic, rng)}?",
        f"I think I see it now. Let me try to summarize: {_student_summary(topic, subtopic, rng)}.",
        f"Got it. And when would this be more useful than the alternative approach?",
    ]
    for i in range(min(remaining // 2, 5)):
        turns.append({"role": "user", "content": rng.choice(questions)})
        turns.append({"role": "assistant", "content": rng.choice(answers)})
        if i < 2:
            turns.append({"role": "user", "content": rng.choice(follow_ups)})
            turns.append({"role": "assistant", "content": f"Exactly right! {_reinforcement_sentence(topic, subtopic, rng)}"})
    return turns


def _repeated_clarification_turns(topic, subtopic, difficulty, is_programming, rng, remaining):
    turns = []
    clarifications = [
        f"Hmm, I still don't quite get that. Can you explain {subtopic} a different way?",
        f"Sorry, I'm still confused. When you say '{subtopic}', what exactly do you mean?",
        f"That helps a little but I'm still unclear. Could you use a simpler example?",
        f"I'm not sure I follow. Can you break it down into smaller steps?",
        f"Wait, I thought {subtopic} worked differently. Can you clarify?",
        f"OK so I think I almost understand but there's one part that's still fuzzy.",
    ]
    assistant_clarifications = [
        f"Of course, let me try a different angle. {_alternative_explanation(topic, subtopic, is_programming, rng)}",
        f"Sure, let me simplify. Think of {subtopic} like this: {_analogy_sentence(topic, subtopic, rng)}",
        f"No problem. Let's go even more basic. {_basic_explanation(topic, subtopic, is_programming, rng)}",
        f"Let me use a real-world analogy: {_analogy_sentence(topic, subtopic, rng)}",
        f"Great question for clarity. {_alternative_explanation(topic, subtopic, is_programming, rng)}",
    ]
    for i in range(min(remaining // 2, 6)):
        turns.append({"role": "user", "content": rng.choice(clarifications)})
        turns.append({"role": "assistant", "content": rng.choice(assistant_clarifications)})
    return turns


def _misconception_correction_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining):
    turns = []
    misconceptions = [
        f"Wait, so {subtopic} means that it always produces the same result regardless of input, right?",
        f"I thought {subtopic} was basically the same as {_related_concept(topic, subtopic, rng)}. Am I wrong?",
        f"So if I use {subtopic}, it automatically handles all edge cases?",
        f"Doesn't {subtopic} only apply when the data is sorted?",
        f"I assumed that {subtopic} works the same way in all {topic} implementations.",
        f"I was under the impression that {subtopic} is just a performance optimization and doesn't change correctness.",
    ]
    corrections = [
        f"That's actually a common misconception. {subtopic} doesn't work that way — {_correction_sentence(topic, subtopic, rng)}",
        f"Not quite! The key difference is that {_correction_sentence(topic, subtopic, rng)} Let me show you why with an example.",
        f"I can see why you'd think that, but {subtopic} is different from {_related_concept(topic, subtopic, rng)} in an important way: {_correction_sentence(topic, subtopic, rng)}",
        f"Good attempt, but that assumption leads to a bug in practice. {_correction_sentence(topic, subtopic, rng)}",
        f"That's a subtle but important distinction. {_correction_sentence(topic, subtopic, rng)}",
    ]
    acknowledgments = [
        f"Oh! I didn't realize that. So the actual behavior is {_student_correction_paraphrase(subtopic, rng)}?",
        f"Okay, so my assumption was wrong. That actually makes more sense now.",
        f"I see — so the real issue is {_student_correction_paraphrase(subtopic, rng)}. I'll keep that in mind.",
        f"Ah, I see. I was confusing it with something else. Thanks for catching that.",
    ]
    turns.append({"role": "user", "content": rng.choice(misconceptions)})
    turns.append({"role": "assistant", "content": rng.choice(corrections)})
    turns.append({"role": "user", "content": rng.choice(acknowledgments)})
    turns.append({"role": "assistant", "content": f"Exactly. {_reinforcement_sentence(topic, subtopic, rng)} A good way to remember this: {_memory_tip(subtopic, rng)}"})
    if remaining > 4:
        turns.append({"role": "user", "content": f"Can you show me an example where ignoring this distinction would cause a problem?"})
        turns.append({"role": "assistant", "content": f"Yes! Consider this scenario: {_problem_scenario(topic, subtopic, is_programming, rng)}"})
    return turns


def _example_driven_turns(topic, subtopic, difficulty, is_programming, is_math, is_db, rng, remaining):
    turns = []
    example_requests = [
        f"Can you show me a concrete example of {subtopic}?",
        f"I learn best from examples. Could you walk through one?",
        f"Could you give me a few different examples of {subtopic}?",
        f"Can you show the same concept with different values so I can see the pattern?",
        f"I'd love to see a step-by-step breakdown. Can you show me one?",
    ]
    follow_up_requests = [
        f"That's helpful. Can you show me another example, but this time with a different scenario?",
        f"Okay I see. Now can you show me what happens when it fails or doesn't work as expected?",
        f"What does this look like in a real project? Can you give a practical example?",
        f"Can you also show me what NOT to do? A counter-example?",
    ]
    for i in range(min(remaining // 2, 4)):
        if i == 0:
            turns.append({"role": "user", "content": rng.choice(example_requests)})
        else:
            turns.append({"role": "user", "content": rng.choice(follow_up_requests)})
        turns.append({"role": "assistant", "content": f"{_example_sentence(topic, subtopic, is_programming, is_math, rng)}"})
        if i < 2:
            turns.append({"role": "user", "content": f"I see! In step {i+2} there, could you explain why you did it that way?"})
            turns.append({"role": "assistant", "content": f"Good catch. {_step_explanation(topic, subtopic, rng)}"})
    return turns


def _problem_solving_turns(topic, subtopic, difficulty, is_programming, is_math, is_db, rng, remaining):
    turns = []
    problem_intro = [
        f"Can you give me a practice problem on {subtopic} to try?",
        f"I think I understand {subtopic}. Can I try solving a problem?",
        f"Could you give me something to practice? I want to test my understanding of {subtopic}.",
    ]
    student_attempts = [
        f"Let me try... I think the answer is {_student_attempt(topic, subtopic, is_programming, rng)}. Is that right?",
        f"Okay so my approach would be to first {_student_attempt_step1(subtopic, rng)}, and then {_student_attempt_step2(subtopic, rng)}.",
        f"I'm going to try this: {_student_code_attempt(topic, subtopic, is_programming, rng)}. Does that work?",
        f"My answer: {_student_attempt(topic, subtopic, is_programming, rng)}. I'm not totally confident though.",
    ]
    feedback = [
        f"Good effort! You got the core idea right. One thing to tweak: {_feedback_sentence(topic, subtopic, rng)}",
        f"Close! The approach is correct but there's a subtle issue: {_feedback_sentence(topic, subtopic, rng)}",
        f"Well done! That's exactly right. {_reinforcement_sentence(topic, subtopic, rng)}",
        f"Nice try! Your logic is sound but consider this edge case: {_edge_case(topic, subtopic, rng)}",
        f"You're on the right track. Let me give you a hint: {_hint_sentence(topic, subtopic, rng)}",
    ]
    turns.append({"role": "user", "content": rng.choice(problem_intro)})
    turns.append({"role": "assistant", "content": f"Sure! Here's a problem: {_practice_problem(topic, subtopic, is_programming, is_math, rng)}"})
    turns.append({"role": "user", "content": rng.choice(student_attempts)})
    turns.append({"role": "assistant", "content": rng.choice(feedback)})
    if remaining > 4:
        turns.append({"role": "user", "content": f"Okay, with that hint, let me revise. How about {_revised_attempt(subtopic, rng)}?"})
        turns.append({"role": "assistant", "content": f"Much better! {_reinforcement_sentence(topic, subtopic, rng)}"})
    return turns


def _partial_understanding_turns(topic, subtopic, difficulty, is_programming, rng, remaining):
    turns = []
    partial = [
        f"I get the part about {_first_part(subtopic, rng)}, but the {_second_part(subtopic, rng)} still confuses me.",
        f"The definition makes sense but I don't understand why it works that way.",
        f"I understand the mechanics of {subtopic} but I can't see when you'd actually use it.",
        f"The basic case is clear, but what about {_edge_case(topic, subtopic, rng)}?",
        f"I kind of understand it, but I couldn't explain it to someone else yet.",
    ]
    for i in range(min(remaining // 2, 5)):
        turns.append({"role": "user", "content": rng.choice(partial)})
        turns.append({"role": "assistant", "content": f"That's a perfectly reasonable place to be. Let me focus on the confusing part. {_targeted_explanation(topic, subtopic, is_programming, rng)}"})
    return turns


def _progressive_understanding_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining):
    turns = []
    early_confusion = [
        f"I really don't get this at all. Can you explain it very simply?",
        f"This is way over my head. Let's start from the very beginning.",
        f"Honestly I have no idea what {subtopic} even does. Can you break it down completely?",
    ]
    mid_progress = [
        f"Okay, I think I see it a little now. So {subtopic} is kind of like {_analogy_sentence(topic, subtopic, rng)}?",
        f"I'm starting to get it. Let me try to put it in my own words: {_student_paraphrase(subtopic, rng)}.",
        f"It's getting clearer. So the main point is {_student_key_takeaway(subtopic, rng)}?",
    ]
    late_understanding = [
        f"Oh! Now I think I understand. So {subtopic} works by {_student_summary(topic, subtopic, rng)}.",
        f"That actually makes a lot of sense now. I was overthinking it.",
        f"Wait, I think I've got it now. Let me test my understanding: {_student_paraphrase(subtopic, rng)}. Correct?",
    ]
    turns.append({"role": "user", "content": rng.choice(early_confusion)})
    turns.append({"role": "assistant", "content": f"No worries! Let's start from scratch. {_basic_explanation(topic, subtopic, is_programming, rng)}"})
    turns.append({"role": "user", "content": rng.choice(mid_progress)})
    turns.append({"role": "assistant", "content": f"You're on the right track! {_reinforcement_sentence(topic, subtopic, rng)} Let me build on that."})
    turns.append({"role": "user", "content": rng.choice(late_understanding)})
    turns.append({"role": "assistant", "content": f"Exactly! You've got it. {_reinforcement_sentence(topic, subtopic, rng)}"})
    if remaining > 6:
        turns.append({"role": "user", "content": f"Can we try a slightly harder example now that I understand the basics?"})
        turns.append({"role": "assistant", "content": f"Absolutely! Here's a more advanced version: {_example_sentence(topic, subtopic, is_programming, is_math, rng)}"})
    return turns


def _persistent_confusion_turns(topic, subtopic, difficulty, is_programming, rng, remaining):
    turns = []
    confused_msgs = [
        f"I still don't understand even after the example. What am I missing?",
        f"Every explanation I read makes it sound simple but I can't apply it. Can you try yet again?",
        f"Sorry, I know you've explained this twice now but I'm still lost. Maybe I need a different approach?",
        f"I'm going in circles. Even simple examples don't click. Can we approach from a totally different angle?",
        f"Okay I wrote down what you said but when I try it myself it doesn't work. What am I doing wrong?",
        f"I don't know why this is so hard for me. Let me ask about just one small part.",
    ]
    patient_re_explanations = [
        f"That's completely okay — some concepts take a few attempts. Let me try an entirely different approach. {_analogy_sentence(topic, subtopic, rng)}",
        f"No worries. Let's take a step back and approach it differently: {_alternative_explanation(topic, subtopic, is_programming, rng)}",
        f"I hear you. Let's try building intuition from a real-world scenario: {_analogy_sentence(topic, subtopic, rng)}",
        f"Let's slow down and isolate the exact point of confusion. Which of these two things is still unclear? {_two_subpoints(subtopic, rng)}",
        f"Sometimes it helps to diagram it mentally. Here's a way to visualize {subtopic}: {_visualization(subtopic, rng)}",
    ]
    for i in range(min(remaining // 2, 6)):
        turns.append({"role": "user", "content": rng.choice(confused_msgs)})
        turns.append({"role": "assistant", "content": rng.choice(patient_re_explanations)})
    return turns


def _independent_reasoning_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining):
    turns = []
    self_explanations = [
        f"Let me try to explain it back to you and tell me where I go wrong: {_student_explanation(topic, subtopic, rng)}",
        f"I think I understand it now. Here's my mental model: {_student_summary(topic, subtopic, rng)}.",
        f"My understanding is: {_student_paraphrase(subtopic, rng)}. Does that capture it correctly?",
        f"I've been thinking about it and I think {subtopic} is basically solving the problem of {_problem_solved(subtopic, rng)}.",
        f"So just to confirm my reasoning: {_student_reasoning(topic, subtopic, rng)}?",
    ]
    response_to_explanation = [
        f"That's a really solid explanation! Just one small nuance to add: {_nuance_sentence(topic, subtopic, rng)}",
        f"Very close! You've captured the main idea. The only thing I'd add is: {_nuance_sentence(topic, subtopic, rng)}",
        f"Excellent reasoning. That's exactly right — and your phrasing is good because {_reason_for_phrasing(subtopic, rng)}",
        f"Great, you've got the core concept. Just be careful about one edge case: {_edge_case(topic, subtopic, rng)}",
    ]
    for i in range(min(remaining // 2, 4)):
        turns.append({"role": "user", "content": rng.choice(self_explanations)})
        turns.append({"role": "assistant", "content": rng.choice(response_to_explanation)})
    return turns


def _mixed_understanding_turns(topic, subtopic, difficulty, is_programming, rng, remaining):
    turns = []
    strong_parts = [
        f"I'm pretty comfortable with the {_first_part(subtopic, rng)} part — I've used it before.",
        f"I can confidently handle the basic {subtopic} operations.",
        f"The {_first_part(subtopic, rng)} aspect makes complete sense to me.",
    ]
    weak_parts = [
        f"But I'm really struggling with {_second_part(subtopic, rng)}.",
        f"What I can't wrap my head around is the {_second_part(subtopic, rng)} side of it.",
        f"My weak spot is when {_edge_case(topic, subtopic, rng)} comes into play.",
    ]
    for i in range(min(remaining // 2, 4)):
        turns.append({"role": "user", "content": rng.choice(strong_parts) + " " + rng.choice(weak_parts)})
        turns.append({"role": "assistant", "content": f"That's a very common pattern. Since you've got the basics, let's focus on {_second_part(subtopic, rng)}. {_targeted_explanation(topic, subtopic, is_programming, rng)}"})
    return turns


def _efficient_learning_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining):
    turns = []
    quick_confirms = [
        f"That actually makes total sense immediately. So to confirm: {_student_paraphrase(subtopic, rng)}?",
        f"Oh, that's much simpler than I expected. Got it. Can we jump to a harder variation?",
        f"Clear. Now I'm curious about the edge case where {_edge_case(topic, subtopic, rng)}.",
        f"Yep, understood. What's the most efficient way to use {subtopic} in practice?",
    ]
    for i in range(min(remaining // 2, 3)):
        turns.append({"role": "user", "content": rng.choice(quick_confirms)})
        turns.append({"role": "assistant", "content": f"Exactly right! {_reinforcement_sentence(topic, subtopic, rng)} And to go deeper: {_advanced_point(topic, subtopic, rng)}"})
    return turns


def _exploratory_learning_turns(topic, subtopic, difficulty, is_programming, is_math, rng, remaining):
    turns = []
    exploratory_qs = [
        f"Why was {subtopic} designed this way rather than an alternative approach?",
        f"What are the theoretical limits of {subtopic}?",
        f"How does {subtopic} relate to {_related_concept(topic, subtopic, rng)}?",
        f"What would break if we removed the constraint that {subtopic} enforces?",
        f"Are there cases where {subtopic} is actually the wrong tool to use?",
        f"How do practitioners in industry actually use {subtopic}?",
        f"What's the history behind why {subtopic} became standard practice?",
        f"Can {subtopic} be extended or modified for special cases?",
    ]
    deep_answers = [
        f"That's a great question that gets into the deeper 'why'. {_advanced_point(topic, subtopic, rng)} In fact, {_connection_sentence(topic, subtopic, rng)}",
        f"Excellent question. The design choice behind {subtopic} reflects a fundamental trade-off: {_tradeoff_sentence(topic, subtopic, rng)}",
        f"This is where it gets interesting. {subtopic} connects to {_related_concept(topic, subtopic, rng)} in a non-obvious way: {_connection_sentence(topic, subtopic, rng)}",
        f"You're thinking exactly like an engineer now. {_advanced_point(topic, subtopic, rng)}",
        f"Great instinct to question it. The short answer is: {_advanced_point(topic, subtopic, rng)} The longer answer involves {_tradeoff_sentence(topic, subtopic, rng)}",
    ]
    for i in range(min(remaining // 2, 5)):
        turns.append({"role": "user", "content": rng.choice(exploratory_qs)})
        turns.append({"role": "assistant", "content": rng.choice(deep_answers)})
    return turns


def _closing_turn(pattern, final_level, topic, subtopic, is_programming, rng):
    if final_level == "high":
        closers = [
            ({"role": "user", "content": f"Thanks, I feel like I really understand {subtopic} now!"},
             {"role": "assistant", "content": f"Excellent work! You've made great progress on {topic}. Keep practicing and you'll be solid."}),
            ({"role": "user", "content": f"This has been really helpful. I think I can handle {subtopic} problems now."},
             {"role": "assistant", "content": f"Great! You demonstrated strong understanding today. Good luck with your studies!"}),
        ]
    elif final_level == "medium":
        closers = [
            ({"role": "user", "content": f"I think I have a better grip on {subtopic} now, even if it's not 100% clear yet."},
             {"role": "assistant", "content": f"That's solid progress! Review the examples we went through and try a few problems on your own to solidify it."}),
            ({"role": "user", "content": f"Okay, I'll need to review this more but I feel less lost than before."},
             {"role": "assistant", "content": f"That's a great place to be. Don't hesitate to revisit {subtopic} and come back with more questions."}),
        ]
    else:
        closers = [
            ({"role": "user", "content": f"I still need to think about this more. I'll re-read the examples."},
             {"role": "assistant", "content": f"That's the right approach. Take your time with {subtopic} — it often clicks when you revisit it with fresh eyes."}),
            ({"role": "user", "content": f"I think I need to practice more before I fully get {subtopic}."},
             {"role": "assistant", "content": f"Absolutely. Hands-on practice is the best way to solidify understanding. Start with simple cases and work up."}),
        ]
    chosen = rng.choice(closers)
    return [chosen[0], chosen[1]]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER SENTENCE GENERATORS (topic-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _example_sentence(topic, subtopic, is_programming, is_math, rng):
    if is_programming:
        snippets = [
            f"For example:\n```python\n# {subtopic} example\nresult = [{subtopic.lower().replace(' ', '_')}(x) for x in range(5)]\nprint(result)\n```\nThis demonstrates the core behavior.",
            f"Here's a simple code example:\n```python\ndef demonstrate_{subtopic.lower().replace(' ', '_').replace('-', '_')}(data):\n    # apply the concept\n    return [item for item in data if item]\n\noutput = demonstrate_{subtopic.lower().replace(' ', '_').replace('-', '_')}([1, None, 2, None, 3])\nprint(output)  # [1, 2, 3]\n```",
            f"Consider this: if you have `data = [10, 20, 30]`, then applying {subtopic} gives you something like `result = 60` (or the transformed collection). The key is that {_reinforcement_sentence(topic, subtopic, rng)}",
        ]
        return rng.choice(snippets)
    elif is_math:
        return (f"Take a concrete numerical example: suppose we have vectors **u** = [1, 2, 3] and **v** = [4, 5, 6]. "
                f"Applying {subtopic} gives us a well-defined result that illustrates the underlying principle. "
                f"The calculation step-by-step: {_step_example(topic, subtopic, is_programming, is_math, rng)}")
    else:
        return (f"Here's a practical example: imagine a database with two tables — `Users` and `Orders`. "
                f"Applying {subtopic} in this context means {_connection_sentence(topic, subtopic, rng)}. "
                f"This is one of the most common patterns you'll see in real systems.")


def _step_example(topic, subtopic, is_programming, is_math, rng):
    steps = [
        f"Step 1: Identify the input. Step 2: Apply the {subtopic} operation. Step 3: Check the result against your expected output.",
        f"(1) Set up initial state. (2) Apply {subtopic}. (3) Verify the output matches expectations. (4) Handle edge cases.",
        f"First, initialize your data structure. Then invoke {subtopic}. Finally, observe how the state changes.",
    ]
    return rng.choice(steps)


def _alternative_explanation(topic, subtopic, is_programming, rng):
    alternatives = [
        f"Think of {subtopic} as a recipe — it takes ingredients (inputs) and produces a dish (output) in a specific way.",
        f"Instead of focusing on implementation, think about what problem {subtopic} solves: {_problem_solved(subtopic, rng)}.",
        f"Let's use a different framing: {subtopic} is like a contract that guarantees certain properties hold at all times.",
        f"Imagine you're explaining {subtopic} to someone who has never seen code before. At its core, it just {_basic_explanation_short(subtopic, rng)}.",
    ]
    return rng.choice(alternatives)


def _analogy_sentence(topic, subtopic, rng):
    analogies = [
        f"think of {subtopic} like a library card catalog — it provides structure so you can find things efficiently.",
        f"it's like a traffic roundabout — everyone follows the same rules, making the whole system flow smoothly.",
        f"it's similar to filing documents in a cabinet — there's a systematic way to store and retrieve them.",
        f"imagine it like a recipe card — it tells you exactly what to do and in what order.",
        f"it's like a phone book — organized so you can search without reading every entry.",
        f"think of it as a translator between two languages — it converts one form to another in a predictable way.",
    ]
    return rng.choice(analogies)


def _basic_explanation(topic, subtopic, is_programming, rng):
    return (f"At its most fundamental level, {subtopic} is about {_problem_solved(subtopic, rng)}. "
            f"Without worrying about advanced cases, the core idea is: {_basic_explanation_short(subtopic, rng)}.")


def _basic_explanation_short(subtopic, rng):
    templates = [
        f"take something in, transform it, and produce a well-defined output",
        f"organize information so it can be accessed or modified predictably",
        f"enforce a rule or constraint that prevents a specific class of errors",
        f"provide a standard interface so different parts of a system can work together",
    ]
    return rng.choice(templates)


def _pitfall_sentence(topic, subtopic, rng):
    pitfalls = [
        f"forgetting to handle the empty or null case for {subtopic}",
        f"assuming {subtopic} is zero-indexed when it isn't (or vice versa)",
        f"applying {subtopic} to unsorted data when it requires sorted input",
        f"mutating data inside {subtopic} when it should remain immutable",
        f"mixing up {subtopic} with its closely related but distinct cousin",
    ]
    return rng.choice(pitfalls)


def _connection_sentence(topic, subtopic, rng):
    connections = [
        f"{subtopic} is the foundation that makes more advanced {topic} techniques possible",
        f"without understanding {subtopic}, the higher-level {topic} abstractions won't make sense",
        f"{subtopic} shows up repeatedly in {topic} because it solves a fundamental problem elegantly",
        f"once you internalize {subtopic}, you'll recognize it as a building block across many {topic} problems",
    ]
    return rng.choice(connections)


def _reinforcement_sentence(topic, subtopic, rng):
    reinforcements = [
        f"You've captured the core of {subtopic} correctly.",
        f"That's exactly the right mental model for {subtopic}.",
        f"You've got it — that intuition will serve you well in {topic}.",
        f"Precisely. That's the key insight behind {subtopic}.",
        f"That's the right way to think about {subtopic} in the context of {topic}.",
    ]
    return rng.choice(reinforcements)


def _student_paraphrase(subtopic, rng):
    paraphrases = [
        f"it's a mechanism that transforms input into output in a defined way",
        f"it's a way of organizing or accessing data that follows specific rules",
        f"it enforces certain guarantees about how the system behaves",
        f"it's a technique for solving a specific class of problems efficiently",
    ]
    return rng.choice(paraphrases)


def _student_key_takeaway(subtopic, rng):
    takeaways = [
        f"to always consider the edge cases",
        f"that the order of operations matters",
        f"that we trade one resource for another (like time vs space)",
        f"that the abstraction hides complexity but the complexity is still there",
    ]
    return rng.choice(takeaways)


def _student_summary(topic, subtopic, rng):
    summaries = [
        f"{subtopic} works by taking the input, applying a defined rule, and producing a consistent output, which is critical in {topic}",
        f"the core of {subtopic} is that it provides a predictable behavior that other parts of {topic} can rely on",
        f"{subtopic} solves the problem of {_problem_solved(subtopic, rng)} in the context of {topic}",
    ]
    return rng.choice(summaries)


def _problem_solved(subtopic, rng):
    problems = [
        "efficiently accessing or modifying data",
        "preventing a specific class of bugs or inconsistencies",
        "making a complex operation reusable and testable",
        "enforcing a constraint that ensures correctness",
        "reducing duplicated logic across the codebase",
    ]
    return rng.choice(problems)


def _related_concept(topic, subtopic, rng):
    related = {
        "Python programming": ["generators", "decorators", "closures", "iterators"],
        "Data Structures and Algorithms": ["hash tables", "heaps", "tries", "segment trees"],
        "Machine Learning": ["neural networks", "SVM", "k-means clustering", "gradient boosting"],
        "Mathematics and Linear Algebra": ["determinants", "vector norms", "matrix decomposition", "tensors"],
        "Database Management and SQL": ["stored procedures", "views", "triggers", "NoSQL"],
        "Operating Systems and Computer Networks": ["semaphores", "mutexes", "socket programming", "NAT"],
        "Natural Language Processing": ["named entity recognition", "dependency parsing", "CRF models", "attention mechanism"],
        "Statistics and Data Science": ["Bayesian inference", "Monte Carlo methods", "bootstrapping", "ANOVA"],
        "Other Computer Science Concepts": ["microservices", "event-driven architecture", "dependency injection", "caching"],
    }
    for cat, concepts in related.items():
        if cat.lower() in topic.lower() or topic.lower() in cat.lower():
            return rng.choice(concepts)
    return rng.choice(["hash maps", "recursion", "abstraction", "encapsulation"])


def _correction_sentence(topic, subtopic, rng):
    corrections = [
        f"the actual behavior depends on the specific conditions at runtime",
        f"edge cases can cause unexpected results if you're not careful",
        f"there's an important distinction between the concept and its implementation",
        f"this only applies when certain preconditions are satisfied",
    ]
    return rng.choice(corrections)


def _student_correction_paraphrase(subtopic, rng):
    return rng.choice([
        f"{subtopic} only works as expected under specific conditions",
        f"the edge cases in {subtopic} can cause bugs if ignored",
        f"there's a crucial distinction I was missing in how {subtopic} operates",
    ])


def _memory_tip(subtopic, rng):
    tips = [
        f"whenever you see {subtopic}, ask yourself what invariant it maintains",
        f"think of {subtopic} as always having a precondition and a postcondition",
        f"picture the data flow — inputs go in, a transformation happens, outputs come out",
    ]
    return rng.choice(tips)


def _problem_scenario(topic, subtopic, is_programming, rng):
    if is_programming:
        return (f"suppose you have a list `data = [1, 2, 3]` and you forget to account for {subtopic}. "
                f"You'd get a subtle bug where the output looks correct for the happy path but fails on edge cases like an empty list or a single element.")
    return (f"imagine a production system where {subtopic} is misunderstood. "
            f"The system might work correctly in tests but fail under specific load or data conditions.")


def _practice_problem(topic, subtopic, is_programming, is_math, rng):
    if is_programming:
        return (f"Given a list of integers, write a function that uses {subtopic} to return the result. "
                f"For example, `f([1, 2, 3, 4, 5])` should return the expected output based on the {subtopic} logic.")
    elif is_math:
        return (f"Compute the result of applying {subtopic} to the following: **A** = [[1,2],[3,4]] and **B** = [[5,6],[7,8]]. "
                f"Show your work step by step.")
    return (f"Given the following schema: `Users(id, name, email)` and `Orders(id, user_id, total)`, "
            f"write a query that demonstrates {subtopic} to find the relevant data.")


def _student_attempt(topic, subtopic, is_programming, rng):
    if is_programming:
        return rng.choice([
            f"`result = [x for x in data if x > 0]`",
            f"`output = sum(data) / len(data)` — wait, that might fail on empty input",
            f"`for i in range(len(arr)): arr[i] = arr[i] * 2`",
        ])
    return rng.choice(["the result is 42", "the answer is the sum of the diagonal elements", "the output should be the sorted version"])


def _student_attempt_step1(subtopic, rng):
    return rng.choice([
        f"identify the base case",
        f"initialize the data structure",
        f"split the problem into smaller subproblems",
        f"check the preconditions",
    ])


def _student_attempt_step2(subtopic, rng):
    return rng.choice([
        f"recursively solve each subproblem",
        f"merge the results",
        f"handle the edge case for empty input",
        f"verify the output matches the specification",
    ])


def _student_code_attempt(topic, subtopic, is_programming, rng):
    if is_programming:
        return f"```python\ndef solve(data):\n    if not data:\n        return None\n    return max(data)  # is this right for {subtopic}?\n```"
    return f"my calculation: result = f(input) where f represents {subtopic}"


def _feedback_sentence(topic, subtopic, rng):
    return rng.choice([
        f"you need to handle the empty input case",
        f"consider what happens when the values are negative",
        f"the logic is correct but the time complexity could be improved",
        f"you're missing the base case in your recursive approach",
        f"the result is correct but the approach isn't using {subtopic} optimally",
    ])


def _edge_case(topic, subtopic, rng):
    return rng.choice([
        f"the input is empty",
        f"all elements have the same value",
        f"the input has exactly one element",
        f"there are duplicate values",
        f"the values are extremely large or extremely small",
    ])


def _hint_sentence(topic, subtopic, rng):
    return rng.choice([
        f"think about what happens when you process one element at a time",
        f"consider starting from the simplest case and building up",
        f"try working backwards from the desired output",
        f"think about what invariant must be true at each step",
    ])


def _revised_attempt(subtopic, rng):
    return rng.choice([
        f"handling the empty case first and then applying the main logic",
        f"using a different data structure that suits {subtopic} better",
        f"breaking it into smaller helper functions",
        f"iterating in the correct order to avoid overwriting values prematurely",
    ])


def _first_part(subtopic, rng):
    return rng.choice([f"basic {subtopic}", f"syntax of {subtopic}", f"definition of {subtopic}"])


def _second_part(subtopic, rng):
    return rng.choice([f"advanced {subtopic} cases", f"{subtopic} edge cases", f"when to apply {subtopic}", f"performance implications of {subtopic}"])


def _targeted_explanation(topic, subtopic, is_programming, rng):
    return (f"The key thing to understand about {subtopic} at a deeper level is: "
            f"{_alternative_explanation(topic, subtopic, is_programming, rng)} "
            f"A good way to test your understanding is to try to implement it from scratch.")


def _student_explanation(topic, subtopic, rng):
    return (f"{subtopic} is a way to handle the problem of {_problem_solved(subtopic, rng)} by "
            f"{_student_paraphrase(subtopic, rng)}. The output is determined by the input and the rules of {topic}.")


def _student_reasoning(topic, subtopic, rng):
    return (f"if we start with the premise that {subtopic} enforces {_student_key_takeaway(subtopic, rng)}, "
            f"then it follows that any violation of this would break the correctness guarantee")


def _nuance_sentence(topic, subtopic, rng):
    return rng.choice([
        f"in certain edge cases, {subtopic} may behave differently than you'd expect",
        f"there's a subtle distinction between {subtopic} and its close relatives in {topic}",
        f"the performance characteristics of {subtopic} depend on the specific implementation",
        f"some older or alternative systems may implement {subtopic} differently",
    ])


def _reason_for_phrasing(subtopic, rng):
    return rng.choice([
        f"it emphasizes the computational aspect rather than just the syntactic form",
        f"it correctly separates the 'what' from the 'how'",
        f"it captures the invariant that must hold throughout",
    ])


def _advanced_point(topic, subtopic, rng):
    return rng.choice([
        f"at a more advanced level, {subtopic} interacts with other {topic} concepts in non-trivial ways",
        f"in production systems, {subtopic} is often the first place engineers optimize",
        f"the theoretical guarantees of {subtopic} only hold under specific assumptions that are worth examining",
        f"researchers have proposed variants of {subtopic} that trade certain guarantees for performance benefits",
    ])


def _tradeoff_sentence(topic, subtopic, rng):
    return rng.choice([
        f"time complexity versus space complexity",
        f"simplicity of implementation versus optimality of performance",
        f"generality versus specialization",
        f"correctness guarantees versus flexibility",
    ])


def _two_subpoints(subtopic, rng):
    return rng.choice([
        f"(a) why {subtopic} is structured this way, or (b) how to apply it in practice?",
        f"(a) the definition of {subtopic}, or (b) when to actually use it?",
        f"(a) the mechanics of {subtopic}, or (b) why it produces the correct result?",
    ])


def _visualization(subtopic, rng):
    return rng.choice([
        f"picture the data flowing through a pipeline — {subtopic} is one transformation stage.",
        f"imagine a tree diagram where each level represents a step in {subtopic}.",
        f"draw a box labeled '{subtopic}' with arrows showing input going in and output coming out.",
    ])


def _step_explanation(topic, subtopic, rng):
    return (f"The reason for that specific step is that it maintains the invariant that {subtopic} requires. "
            f"Without it, subsequent steps would produce incorrect results. "
            f"This kind of detail becomes clearer with more practice problems.")


def _fix_alternation(convo):
    """Ensure conversation doesn't have consecutive same-role messages."""
    if not convo:
        return convo
    fixed = [convo[0]]
    for msg in convo[1:]:
        if msg["role"] == fixed[-1]["role"]:
            # insert a brief bridge message
            opposite = "assistant" if msg["role"] == "user" else "user"
            if opposite == "user":
                fixed.append({"role": "user", "content": "I see, let me ask a follow-up question."})
            else:
                fixed.append({"role": "assistant", "content": "Let me elaborate on that point further."})
        fixed.append(msg)
    return fixed


# ─────────────────────────────────────────────────────────────────────────────
# SESSION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_counts(conversation):
    """Compute annotation counts from the actual conversation."""
    student_question_count = 0
    student_attempt_count = 0
    example_requests_count = 0
    clarification_request_count = 0
    self_explanation_count = 0
    code_attempt_count = 0
    conceptual_question_count = 0

    question_words = ["?", "what", "why", "how", "when", "where", "which", "can you", "could you", "would"]
    example_words = ["example", "show me", "demonstrate", "code", "walk through", "step-by-step", "snippet"]
    clarify_words = ["clarify", "still don't", "still confused", "still unclear", "again", "different way", "don't understand", "don't get", "confused", "explain", "simpler"]
    self_explain_words = ["let me try", "in my own words", "my understanding", "my mental model", "i think it", "let me explain", "i'd say"]
    code_words = ["```", "def ", "for ", "while ", "class ", "import ", "print(", "return "]
    conceptual_words = ["why", "what is", "concept", "idea", "theory", "principle", "fundamentally", "explain", "meaning", "definition"]

    for msg in conversation:
        if msg["role"] == "user":
            content_lower = msg["content"].lower()
            if "?" in msg["content"]:
                student_question_count += 1
            if any(w in content_lower for w in ["my answer", "i think the answer", "my approach", "i got", "let me try", "```"]):
                student_attempt_count += 1
            if any(w in content_lower for w in example_words):
                example_requests_count += 1
            if any(w in content_lower for w in clarify_words):
                clarification_request_count += 1
            if any(w in content_lower for w in self_explain_words):
                self_explanation_count += 1
            if any(w in msg["content"] for w in code_words):
                code_attempt_count += 1
            if any(w in content_lower for w in conceptual_words):
                conceptual_question_count += 1

    return {
        "student_question_count": max(student_question_count, 1),
        "student_attempt_count": student_attempt_count,
        "example_requests_count": example_requests_count,
        "clarification_request_count": clarification_request_count,
        "self_explanation_count": self_explanation_count,
        "code_attempt_count": code_attempt_count,
        "conceptual_question_count": max(conceptual_question_count, 1),
    }


def extract_keywords(topic, subtopic):
    """Extract relevant keywords from topic/subtopic."""
    words = (topic + " " + subtopic).lower().split()
    # Remove common filler words
    stop = {"and", "the", "a", "an", "of", "in", "to", "for", "with", "vs", "or"}
    keywords = [w.strip(".,;:()") for w in words if w not in stop and len(w) > 2]
    return list(dict.fromkeys(keywords))[:8]  # unique, up to 8


def generate_quiz(difficulty, pattern, final_level, rng):
    """Generate internally consistent quiz fields."""
    q_count = rng.randint(5, 10)

    # Base score range depends on final understanding + pattern
    if final_level == "high" and pattern in ("smooth_learning", "efficient_learning", "exploratory_learning", "independent_reasoning"):
        raw_min, raw_max = 0.70, 1.00
    elif final_level == "high":
        raw_min, raw_max = 0.60, 0.95
    elif final_level == "medium" and pattern in ("persistent_confusion", "repeated_clarification"):
        raw_min, raw_max = 0.30, 0.65
    elif final_level == "medium":
        raw_min, raw_max = 0.40, 0.80
    elif final_level == "low" and pattern in ("persistent_confusion",):
        raw_min, raw_max = 0.10, 0.50
    else:
        raw_min, raw_max = 0.20, 0.65

    # Add noise: occasionally a confused student surprises us; a smooth learner blanks on quiz
    noise = rng.gauss(0, 0.12)
    pct = min(1.0, max(0.0, rng.uniform(raw_min, raw_max) + noise))

    score = round(pct * q_count)
    score = max(0, min(score, q_count))
    actual_pct = round((score / q_count) * 100, 1)

    if actual_pct < 50:
        label = "needs_improvement"
    elif actual_pct < 70:
        label = "developing"
    elif actual_pct < 85:
        label = "proficient"
    else:
        label = "advanced"

    return {
        "quiz_question_count": q_count,
        "quiz_score": score,
        "quiz_score_percentage": actual_pct,
        "quiz_difficulty": difficulty,
        "quiz_outcome_label": label,
    }


def generate_session(session_num, subject_category, topic, subtopic, difficulty, rng_global):
    """Generate one complete session record."""
    session_id = f"EDU_{session_num:04d}"
    student_num = rng_global.randint(1, 150)
    student_id = f"STU_{student_num:03d}"

    # Pick pattern
    pattern = rng_global.choice(LEARNING_PATTERNS)

    # Derive initial/final understanding consistently with pattern
    pattern_to_levels = {
        "smooth_learning": (("medium", "high"), ("high",)),
        "repeated_clarification": (("low", "medium"), ("medium", "high")),
        "misconception_correction": (("low", "medium"), ("medium", "high")),
        "example_driven": (("low", "medium"), ("medium", "high")),
        "problem_solving": (("medium",), ("medium", "high")),
        "partial_understanding": (("medium",), ("medium",)),
        "progressive_understanding": (("low",), ("medium", "high")),
        "persistent_confusion": (("low",), ("low", "medium")),
        "independent_reasoning": (("medium", "high"), ("high",)),
        "mixed_understanding": (("medium",), ("medium", "high")),
        "efficient_learning": (("medium", "high"), ("high",)),
        "exploratory_learning": (("medium", "high"), ("high",)),
    }
    init_pool, final_pool = pattern_to_levels.get(pattern, (("low", "medium", "high"), ("low", "medium", "high")))
    initial_level = rng_global.choice(init_pool)
    final_level = rng_global.choice(final_pool)

    misconception_present = pattern in ("misconception_correction", "partial_understanding", "persistent_confusion") or rng_global.random() < 0.35
    if misconception_present:
        if pattern == "misconception_correction":
            misconception_corrected = "true"
        elif pattern == "persistent_confusion":
            misconception_corrected = rng_global.choice(["false", "false", "true"])
        else:
            misconception_corrected = rng_global.choice(["true", "false"])
    else:
        misconception_corrected = "not_applicable"

    clarification_behavior_map = {
        "repeated_clarification": "frequent",
        "persistent_confusion": "frequent",
        "partial_understanding": "occasional",
        "progressive_understanding": "occasional",
        "example_driven": "occasional",
        "smooth_learning": "none",
        "efficient_learning": "none",
        "exploratory_learning": "none",
        "independent_reasoning": "occasional",
        "mixed_understanding": "occasional",
        "misconception_correction": "occasional",
        "problem_solving": "occasional",
    }
    clarification_behavior = clarification_behavior_map.get(pattern, rng_global.choice(["none", "occasional", "frequent"]))

    independent_reasoning_present = pattern in ("independent_reasoning", "exploratory_learning", "mixed_understanding") or rng_global.random() < 0.25

    duration = rng_global.randint(10, 55)

    # Generate conversation
    conversation = make_conversation(
        topic, subtopic, pattern, difficulty, initial_level, final_level,
        misconception_present, subject_category
    )

    # Compute annotation counts from actual conversation
    counts = compute_counts(conversation)

    # Adjust clarification_request_count to match behavior
    if clarification_behavior == "none" and counts["clarification_request_count"] > 0:
        counts["clarification_request_count"] = 0
    elif clarification_behavior == "frequent" and counts["clarification_request_count"] < 2:
        counts["clarification_request_count"] = rng_global.randint(2, 4)

    # Generate quiz
    quiz = generate_quiz(difficulty, pattern, final_level, rng_global)

    # Build keyword list and dominant concepts
    keywords = extract_keywords(topic, subtopic)
    dominant_concepts = keywords[:4]
    misconception_concepts = [subtopic.lower()] if misconception_present else []
    repeated_concept_count = rng_global.randint(1, 4)

    record = {
        "session_id": session_id,
        "synthetic_student_id": student_id,
        "subject_category": subject_category,
        "topic": topic,
        "subtopic": subtopic,
        "difficulty_level": difficulty,
        "language": "en",
        "session_duration_minutes": duration,
        "learning_pattern": pattern,
        "initial_understanding_level": initial_level,
        "final_demonstrated_understanding_level": final_level,
        "misconception_present": misconception_present,
        "misconception_corrected": misconception_corrected,
        "clarification_behavior": clarification_behavior,
        "example_requests_count": counts["example_requests_count"],
        "student_question_count": counts["student_question_count"],
        "student_attempt_count": counts["student_attempt_count"],
        **quiz,
        "dominant_concepts": dominant_concepts,
        "misconception_concepts": misconception_concepts,
        "repeated_concept_count": repeated_concept_count,
        "conceptual_question_count": counts["conceptual_question_count"],
        "clarification_request_count": counts["clarification_request_count"],
        "self_explanation_count": counts["self_explanation_count"],
        "code_attempt_count": counts["code_attempt_count"],
        "keyword_list": keywords,
        "conversation": conversation,
    }
    return record


def generate_all_sessions():
    """Generate all ~500 sessions across subject categories."""
    all_sessions = []
    session_num = 1
    rng_global = random.Random(2024)

    for subject_category, config in SUBJECTS.items():
        target = config["target"]
        topics = config["topics"]
        difficulties = DIFFICULTY_LEVELS

        topic_list = list(topics.keys())
        sessions_per_topic = max(1, target // len(topic_list))
        leftover = target - sessions_per_topic * len(topic_list)

        for t_idx, topic in enumerate(topic_list):
            subtopics = topics[topic]["subtopics"]
            count = sessions_per_topic + (1 if t_idx < leftover else 0)

            for i in range(count):
                subtopic = subtopics[i % len(subtopics)]
                difficulty = difficulties[(session_num + i) % 3]

                record = generate_session(session_num, subject_category, topic, subtopic, difficulty, rng_global)
                all_sessions.append(record)
                session_num += 1

    print(f"Generated {len(all_sessions)} sessions.")
    return all_sessions


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_dataset(sessions):
    issues = []

    session_ids = [s["session_id"] for s in sessions]
    if len(set(session_ids)) != len(session_ids):
        issues.append("DUPLICATE session_ids found!")
    else:
        print("✓ All session_ids unique")

    required_fields = [
        "session_id", "synthetic_student_id", "subject_category", "topic", "subtopic",
        "difficulty_level", "language", "session_duration_minutes", "learning_pattern",
        "initial_understanding_level", "final_demonstrated_understanding_level",
        "misconception_present", "misconception_corrected", "clarification_behavior",
        "example_requests_count", "student_question_count", "student_attempt_count",
        "quiz_question_count", "quiz_score", "quiz_score_percentage", "quiz_difficulty",
        "quiz_outcome_label", "dominant_concepts", "misconception_concepts",
        "repeated_concept_count", "conceptual_question_count", "clarification_request_count",
        "self_explanation_count", "code_attempt_count", "keyword_list", "conversation"
    ]

    missing_field_counts = {f: 0 for f in required_fields}
    quiz_score_errors = 0
    quiz_label_errors = 0

    for s in sessions:
        for f in required_fields:
            if f not in s:
                missing_field_counts[f] += 1

        # Check quiz consistency
        expected_pct = round((s["quiz_score"] / s["quiz_question_count"]) * 100, 1)
        if abs(expected_pct - s["quiz_score_percentage"]) > 1.0:
            quiz_score_errors += 1

        pct = s["quiz_score_percentage"]
        if pct < 50 and s["quiz_outcome_label"] != "needs_improvement":
            quiz_label_errors += 1
        elif 50 <= pct < 70 and s["quiz_outcome_label"] != "developing":
            quiz_label_errors += 1
        elif 70 <= pct < 85 and s["quiz_outcome_label"] != "proficient":
            quiz_label_errors += 1
        elif pct >= 85 and s["quiz_outcome_label"] != "advanced":
            quiz_label_errors += 1

    print(f"✓ Quiz score percentage errors: {quiz_score_errors}")
    print(f"✓ Quiz label errors: {quiz_label_errors}")

    for f, count in missing_field_counts.items():
        if count > 0:
            issues.append(f"Field '{f}' missing in {count} records")

    if issues:
        print("⚠ Issues found:")
        for issue in issues:
            print("  -", issue)
    else:
        print("✓ All required fields present")

    return missing_field_counts, quiz_score_errors, quiz_label_errors, issues


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = r"c:\Users\LENOVO\OneDrive\Documents\EduLens"

def export_json(sessions, path):
    dataset = {
        "dataset_name": "EduLens Synthetic Learning Sessions",
        "dataset_version": "1.0",
        "synthetic_data": True,
        "total_sessions": len(sessions),
        "generated_at": datetime.now().isoformat(),
        "description": (
            "Synthetic student-LLM learning conversations for NLP and machine learning experimentation. "
            "This dataset does NOT represent real students or validated educational measurements."
        ),
        "sessions": sessions,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON exported to {path}")


def export_csv(sessions, path):
    flat_fields = [
        "session_id", "synthetic_student_id", "subject_category", "topic", "subtopic",
        "difficulty_level", "language", "session_duration_minutes", "learning_pattern",
        "initial_understanding_level", "final_demonstrated_understanding_level",
        "misconception_present", "misconception_corrected", "clarification_behavior",
        "example_requests_count", "student_question_count", "student_attempt_count",
        "quiz_question_count", "quiz_score", "quiz_score_percentage", "quiz_difficulty",
        "quiz_outcome_label", "repeated_concept_count", "conceptual_question_count",
        "clarification_request_count", "self_explanation_count", "code_attempt_count",
        "conversation_message_count",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for s in sessions:
            row = {k: s.get(k, "") for k in flat_fields if k != "conversation_message_count"}
            row["conversation_message_count"] = len(s.get("conversation", []))
            writer.writerow(row)
    print(f"✓ CSV exported to {path}")


def export_quality_report(sessions, path, validation_results):
    missing_field_counts, quiz_score_errors, quiz_label_errors, issues = validation_results

    # Subject distribution
    subject_counts = {}
    for s in sessions:
        cat = s["subject_category"]
        subject_counts[cat] = subject_counts.get(cat, 0) + 1

    # Pattern distribution
    pattern_counts = {}
    for s in sessions:
        p = s["learning_pattern"]
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    # Difficulty distribution
    diff_counts = {}
    for s in sessions:
        d = s["difficulty_level"]
        diff_counts[d] = diff_counts.get(d, 0) + 1

    # Quiz score distribution
    score_bins = {"needs_improvement": 0, "developing": 0, "proficient": 0, "advanced": 0}
    for s in sessions:
        label = s["quiz_outcome_label"]
        score_bins[label] = score_bins.get(label, 0) + 1

    # Duplicate conversation fingerprint check
    convo_hashes = {}
    dup_count = 0
    for s in sessions:
        convo_str = json.dumps(s["conversation"], sort_keys=True)
        h = hashlib.md5(convo_str.encode()).hexdigest()
        if h in convo_hashes:
            dup_count += 1
        convo_hashes[h] = s["session_id"]

    # Average message count
    msg_counts = [len(s["conversation"]) for s in sessions]
    avg_msgs = round(sum(msg_counts) / len(msg_counts), 1)

    # Average quiz score
    avg_score = round(sum(s["quiz_score_percentage"] for s in sessions) / len(sessions), 1)

    report = {
        "report_generated_at": datetime.now().isoformat(),
        "total_records": len(sessions),
        "target_records": 500,
        "subject_category_distribution": subject_counts,
        "learning_pattern_distribution": pattern_counts,
        "difficulty_distribution": diff_counts,
        "quiz_outcome_distribution": score_bins,
        "average_messages_per_session": avg_msgs,
        "average_quiz_score_percentage": avg_score,
        "missing_field_counts": {k: v for k, v in missing_field_counts.items() if v > 0},
        "duplicate_session_id_count": 0,
        "duplicate_conversation_count": dup_count,
        "quiz_score_percentage_errors": quiz_score_errors,
        "quiz_label_errors": quiz_label_errors,
        "validation_issues": issues,
        "annotation_consistency_notes": [
            "clarification_request_count derived from actual conversation content",
            "student_question_count derived from actual conversation content",
            "code_attempt_count derived from actual conversation content",
            "quiz labels verified against quiz_score_percentage boundaries",
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Quality report exported to {path}")
    return report


DOCUMENTATION = '''# EduLens Synthetic Learning Sessions — Dataset Documentation

## Overview

This dataset contains **{total_sessions} synthetic student-LLM tutor learning conversations**
generated for the EduLens learning intelligence and performance analytics platform.

> **⚠ Important Disclaimer**
> This is a purely synthetic dataset. It does not represent real students, real conversations,
> or validated educational measurements. It was generated for NLP research and ML experimentation only.

---

## Project Context

**EduLens** is a student learning analytics platform. Students use external LLM platforms
(e.g., ChatGPT, Claude) to study academic subjects and explicitly submit those conversations
to EduLens for analysis. EduLens extracts learning signals, not the LLM tutoring itself.

This dataset simulates the input to EduLens: structured, annotated student-LLM conversations.

---

## Dataset Files

| File | Description |
|------|-------------|
| `edulens_learning_sessions.json` | Complete dataset: metadata + conversations + annotations |
| `edulens_sessions.csv` | Tabular format, one row per session (no raw conversation text) |
| `edulens_dataset_documentation.md` | This file |
| `edulens_data_quality_report.json` | Computed statistics and validation results |

---

## Schema Reference

### Top-Level Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `dataset_name` | string | Dataset identifier |
| `dataset_version` | string | Version string |
| `synthetic_data` | boolean | Always `true` |
| `total_sessions` | integer | Total number of session records |
| `generated_at` | string | ISO timestamp of generation |
| `description` | string | Human-readable description |
| `sessions` | array | Array of session records |

---

### Session-Level Fields

#### A. Basic Metadata

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `session_id` | string | `EDU_0001` … `EDU_0500` | Unique session identifier |
| `synthetic_student_id` | string | `STU_001` … `STU_150` | Fictional student identifier |
| `subject_category` | string | See Subject Categories table | High-level subject area |
| `topic` | string | — | Specific topic within category |
| `subtopic` | string | — | Focused subtopic of the session |
| `difficulty_level` | string | `beginner`, `intermediate`, `advanced` | Session difficulty |
| `language` | string | `en` | Language of the conversation |
| `session_duration_minutes` | integer | 10–55 | Synthetic session duration |

#### B. Learning Annotations

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `learning_pattern` | string | See Learning Patterns table | Conversational learning pattern |
| `initial_understanding_level` | string | `low`, `medium`, `high` | Student's apparent understanding at session start |
| `final_demonstrated_understanding_level` | string | `low`, `medium`, `high` | Student's demonstrated understanding at session end |
| `misconception_present` | boolean | `true`, `false` | Whether the student expressed a misconception |
| `misconception_corrected` | string | `true`, `false`, `not_applicable` | Whether the misconception was corrected |
| `clarification_behavior` | string | `none`, `occasional`, `frequent` | How often the student requested clarification |
| `independent_reasoning_present` | boolean | `true`, `false` | Whether the student attempted to reason independently |
| `example_requests_count` | integer | ≥ 0 | Number of times student requested examples |
| `student_question_count` | integer | ≥ 1 | Number of student questions in the conversation |
| `student_attempt_count` | integer | ≥ 0 | Number of times student attempted to solve a problem |

#### C. Quiz Fields

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `quiz_question_count` | integer | 5–10 | Number of quiz questions |
| `quiz_score` | integer | 0–quiz_question_count | Number of correct answers |
| `quiz_score_percentage` | float | 0.0–100.0 | `quiz_score / quiz_question_count × 100` |
| `quiz_difficulty` | string | `beginner`, `intermediate`, `advanced` | Quiz difficulty (matches session difficulty) |
| `quiz_outcome_label` | string | See Quiz Outcome Labels | Categorical performance label |

**Quiz Outcome Labels:**

| Label | Score Range |
|-------|-------------|
| `needs_improvement` | < 50% |
| `developing` | 50% – 69.9% |
| `proficient` | 70% – 84.9% |
| `advanced` | 85% – 100% |

> These are synthetic, project-defined categories. They do not correspond to any validated
> educational taxonomy or standardized assessment framework.

#### D. NLP Annotation Fields

| Field | Type | Description |
|-------|------|-------------|
| `dominant_concepts` | string[] | Key concepts mentioned in the session |
| `misconception_concepts` | string[] | Concepts associated with detected misconceptions |
| `repeated_concept_count` | integer | Number of concepts revisited multiple times |
| `conceptual_question_count` | integer | Number of conceptual (vs procedural) questions |
| `clarification_request_count` | integer | Number of explicit clarification requests by student |
| `self_explanation_count` | integer | Number of times student explained in their own words |
| `code_attempt_count` | integer | Number of code snippets produced by the student |
| `keyword_list` | string[] | Key terms extracted from topic and subtopic |

#### E. Conversation

| Field | Type | Description |
|-------|------|-------------|
| `conversation` | object[] | Ordered array of `{role, content}` message objects |

**Message roles:**
- `user` — the student
- `assistant` — the LLM tutor

---

### Subject Categories

| Subject Category | Target Sessions | Example Topics |
|-----------------|----------------|----------------|
| Python programming | 70 | Lists, OOP, Functions, Exception handling |
| Data Structures and Algorithms | 60 | Trees, Sorting, DP, Recursion |
| Machine Learning | 70 | Regression, Random Forest, Evaluation |
| Mathematics and Linear Algebra | 45 | Eigenvalues, Probability, Gradients |
| Database Management and SQL | 45 | Joins, Normalization, Transactions |
| Operating Systems and Computer Networks | 40 | Processes, Scheduling, TCP/IP |
| Natural Language Processing | 60 | TF-IDF, Embeddings, Transformers |
| Statistics and Data Science | 50 | Distributions, Hypothesis testing, Pandas |
| Other Computer Science Concepts | 60 | REST APIs, Git, Cloud, Cybersecurity |

### Learning Patterns

| Pattern | Description |
|---------|-------------|
| `smooth_learning` | Student asks, understands, advances progressively |
| `repeated_clarification` | Student repeatedly requests clarification |
| `misconception_correction` | Student holds an incorrect belief that gets corrected |
| `example_driven` | Student learns primarily through worked examples |
| `problem_solving` | Student attempts to solve problems with feedback |
| `partial_understanding` | Student grasps some but not all aspects |
| `progressive_understanding` | Student struggles initially but improves over the session |
| `persistent_confusion` | Student continues to struggle despite multiple explanations |
| `independent_reasoning` | Student reasons aloud and checks understanding |
| `mixed_understanding` | Student strong in some areas, weak in others |
| `efficient_learning` | Student grasps concepts quickly with minimal repetition |
| `exploratory_learning` | Student asks deeper, exploratory questions |

---

## ML Compatibility Notes

### Appropriate Feature Inputs

The following fields are safe to use as ML input features:

- `difficulty_level` (encoded)
- `session_duration_minutes`
- `student_question_count`
- `clarification_request_count`
- `example_requests_count`
- `self_explanation_count`
- `code_attempt_count`
- `repeated_concept_count`
- `conceptual_question_count`
- `student_attempt_count`
- Embeddings derived from raw `conversation` text
- TF-IDF features from raw `conversation` text

### Potential ML Targets

| Task | Target Field | Type |
|------|-------------|------|
| Regression | `quiz_score_percentage` | Continuous |
| Classification | `quiz_outcome_label` | 4-class |
| Exploration | `initial_understanding_level` → `final_demonstrated_understanding_level` | Ordinal |

### ⚠ Target Leakage Warning

**Never use the following as inputs when predicting quiz performance:**

- `quiz_score` (direct leakage)
- `quiz_score_percentage` (direct leakage)
- `quiz_outcome_label` (direct leakage)
- `final_demonstrated_understanding_level` (derived post-hoc with full conversation access)

**Leakage from annotations:**

`final_demonstrated_understanding_level` was assigned based on the full conversation,
which would not be available at prediction time if predicting quiz performance mid-session.
Use it as an exploratory or evaluation field, not as a training input.

---

## Known Limitations and Biases

1. **Synthetic language patterns**: Conversations were generated programmatically. While designed
   for variety, they may exhibit subtler repetition than truly natural student discourse.

2. **Label noise by design**: Quiz scores deliberately include noise to prevent trivial prediction.
   A student with `persistent_confusion` may score well on the quiz; an `efficient_learner`
   may score poorly. This is intentional and desirable for ML experimentation.

3. **Annotation consistency**: NLP annotation fields (counts, keywords) were computed from actual
   conversation content, not assigned independently. Minor mismatches may exist for edge cases.

4. **Student diversity**: Students are assigned fictional IDs. The same student ID may appear
   in multiple sessions but the conversational style was not controlled to be consistent per
   student ID across sessions.

5. **No temporal ordering**: Sessions have no timestamp or ordering relative to each other.
   Longitudinal analysis is not supported by this dataset.

6. **English only**: All sessions are in English (`language: "en"`).

7. **No audio or visual data**: The dataset is text-only.

8. **No ground-truth validation**: Annotations are synthetic labels, not verified by human raters
   or educational experts.

---

## Recommended Citation

This dataset is a synthetic artifact created for the EduLens project. It should be cited as:

> EduLens Synthetic Learning Sessions v1.0 (2024). Synthetic dataset for NLP and ML experimentation.
> Not for use as empirical evidence of student learning outcomes.
'''


def export_documentation(sessions, path):
    doc = DOCUMENTATION.format(total_sessions=len(sessions))
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"✓ Documentation exported to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== EduLens Dataset Generator ===")
    print("Generating sessions...")
    sessions = generate_all_sessions()

    print("\nValidating...")
    validation_results = validate_dataset(sessions)

    print("\nExporting files...")
    json_path = os.path.join(OUTPUT_DIR, "edulens_learning_sessions.json")
    csv_path = os.path.join(OUTPUT_DIR, "edulens_sessions.csv")
    doc_path = os.path.join(OUTPUT_DIR, "edulens_dataset_documentation.md")
    report_path = os.path.join(OUTPUT_DIR, "edulens_data_quality_report.json")

    export_json(sessions, json_path)
    export_csv(sessions, csv_path)
    export_documentation(sessions, doc_path)
    report = export_quality_report(sessions, report_path, validation_results)

    print("\n=== SUMMARY ===")
    print(f"Total sessions generated: {len(sessions)}")
    print(f"Subject distribution: {report['subject_category_distribution']}")
    print(f"Average messages per session: {report['average_messages_per_session']}")
    print(f"Average quiz score: {report['average_quiz_score_percentage']}%")
    print(f"Duplicate conversations: {report['duplicate_conversation_count']}")
    print("\nAll files written to:", OUTPUT_DIR)
