# Intro to Data Structures

---

### 1. What are the different types of data structures in Python?

Python offers a range of built-in data structures, and you can also create more complex, user-defined ones. They are broadly categorized as follows:

**Built-in Data Structures:**

* **Lists:** Ordered, mutable (changeable) collections of items that can be of different data types. They are defined with square brackets `[]`.
* **Tuples:** Ordered, immutable (unchangeable) collections of items. They are defined with parentheses `()` and are generally faster than lists.
* **Dictionaries:** Unordered collections of key-value pairs. They are mutable and are defined with curly braces `{}`. Keys must be unique and immutable.
* **Sets:** Unordered, mutable collections of unique elements. They are defined with curly braces `{}` and are highly optimized for membership testing.

**User-Defined Data Structures:**

These are typically implemented using classes and objects.

* **Stack:** A linear data structure that follows the Last-In, First-Out (LIFO) principle.
* **Queue:** A linear data structure that follows the First-In, First-Out (FIFO) principle.
* **Tree:** A hierarchical structure with a root node and child nodes.
* **Linked List:** A linear data structure where elements are linked using pointers.
* **Graph:** A non-linear data structure consisting of nodes (vertices) and edges.

---

### 2. What is recursion? How to implement Fibonacci series?

**Recursion** is a programming concept where a function calls itself in order to solve a problem. A recursive function must have two key parts:

1.  **Base Case:** A condition that stops the recursion. Without a base case, the function would call itself infinitely, leading to a "stack overflow" error.
2.  **Recursive Step:** The part of the function where it calls itself, typically with a modified argument that moves it closer to the base case.

Think of it like a set of Russian nesting dolls; you keep opening dolls (the recursive step) until you reach the smallest one that can't be opened (the base case).

**Fibonacci Series using Recursion:**

The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, usually starting with 0 and 1. (0, 1, 1, 2, 3, 5, 8, ...)

Here is a Python function to generate the nth Fibonacci number using recursion:

```python
def fibonacci(n):
    # Base case: The first two numbers in the sequence
    if n <= 1:
        return n
    # Recursive step: Sum of the two preceding numbers
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Example: Get the 10th Fibonacci number
n = 10
print(f"The {n}th Fibonacci number is: {fibonacci(n)}")
