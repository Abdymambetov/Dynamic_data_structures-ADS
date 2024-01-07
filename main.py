import json
from datetime import datetime


# Задание 1
def process_file(file_path):
    even_list = []
    odd_list = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                numbers = map(int, line.split())
                for num in numbers:
                    if num % 2==0:
                        even_list.append(num)
                    else: odd_list.append(num)
    except IOError as e:
        print('Error reading the file: ', str(e))
        return None, None
    return  even_list,  odd_list

file_path = 'example.txt'

even_numbers, odd_numbers = process_file(file_path)

if even_numbers is not None and odd_numbers is not None:
    print('List of even numbers: ', even_numbers)
    print('List of odd numbers: ', odd_numbers)




# Задание 2
class Polynomial:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    @classmethod
    def input_polynomial(cls):
        coefficients = list(map(float, input("Введите коэффициенты полинома через пробел: ").split()))
        return cls(coefficients)
    
    def print_polynomial(self):
        terms = []
        for i, coeff in enumerate(self.coefficients[::-1]):
            if coeff != 0:
                term = f"{coeff}x^{len(self.coefficients)-1-i}"
                terms.append(term)
        polynomial_str = " + ".join(terms) if terms else "0"
        print(polynomial_str)
    
    def evaluate_at(self, x):
        result = sum(coeff * (x ** (len(self.coefficients) - 1 - i)) for i, coeff in enumerate(self.coefficients))
        return result
    
    def add_polynomials(self, other):
        max_len = max(len(self.coefficients), len(other.coefficients))
        new_coefficients = [0] * max_len

        for i in range(len(self.coefficients)):
            new_coefficients[i] += self.coefficients[i]

        for i in range(len(other.coefficients)):
            new_coefficients[i] += other.coefficients[i]

        return Polynomial(new_coefficients)
    
    def multiply_polynomials(self, other):
        result_deg = len(self.coefficients) + len(other.coefficients) - 2
        result_coefficients = [0] * (result_deg + 1)

        for i, coeff1 in enumerate(self.coefficients):
            for j, coeff2 in enumerate(other.coefficients):
                result_coefficients[i + j] += coeff1 * coeff2

        return Polynomial(result_coefficients)

    def differentiate(self):
        result_coefficients = [i * coeff for i, coeff in enumerate(self.coefficients)][1:]
        return Polynomial(result_coefficients)

    def integrate(self, constant=0):
        result_coefficients = [constant] + [(coeff / (i + 1)) for i, coeff in enumerate(self.coefficients)]
        return Polynomial(result_coefficients)


poly1 = Polynomial.input_polynomial()
poly2 = Polynomial.input_polynomial()

print("\nПолином 1:")
poly1.print_polynomial()

print("\nПолином 2:")
poly2.print_polynomial()

point = float(input("\nВведите точку для вычисления полинома: "))
result = poly1.evaluate_at(point)
print(f"Значение полинома в точке {point}: {result}")

sum_poly = poly1.add_polynomials(poly2)
print("\nСумма полиномов:")
sum_poly.print_polynomial()

product_poly = poly1.multiply_polynomials(poly2)
print("\nПроизведение полиномов:")
product_poly.print_polynomial()

derivative_poly = poly1.differentiate()
print("\nПроизводная полинома 1:")
derivative_poly.print_polynomial()

integral_poly = poly1.integrate()
print("\nПервообразная полинома 1:")
integral_poly.print_polynomial()


# Задание 3
class SparseMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.elements = {}  

    @classmethod
    def input_matrix(cls):
        rows = int(input("Введите количество строк матрицы: "))
        cols = int(input("Введите количество столбцов матрицы: "))

        matrix = cls(rows, cols)

        non_zero_count = int(input("Введите количество ненулевых элементов: "))
        for _ in range(non_zero_count):
            row = int(input("Введите номер строки (от 0 до {}): ".format(rows - 1)))
            col = int(input("Введите номер столбца (от 0 до {}): ".format(cols - 1)))
            value = float(input("Введите значение элемента: "))
            matrix.set_element(row, col, value)

        return matrix

    def print_matrix(self):
        for row in range(self.rows):
            for col in range(self.cols):
                print(self.get_element(row, col), end=" ")
            print()

    def set_element(self, row, col, value):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if value != 0:
                self.elements[(row, col)] = value
            elif (row, col) in self.elements:
                del self.elements[(row, col)]

    def get_element(self, row, col):
        return self.elements.get((row, col), 0)

    def add_matrices(self, other):
        result = SparseMatrix(self.rows, self.cols)

        for (row, col), value in self.elements.items():
            result.set_element(row, col, value)

        for (row, col), value in other.elements.items():
            result.set_element(row, col, result.get_element(row, col) + value)

        return result

    def multiply_matrices(self, other):
        if self.cols != other.rows:
            raise ValueError("Нельзя умножить матрицы с такими размерами")

        result = SparseMatrix(self.rows, other.cols)

        for i in range(self.rows):
            for j in range(other.cols):
                dot_product = sum(self.get_element(i, k) * other.get_element(k, j) for k in range(self.cols))
                result.set_element(i, j, dot_product)

        return result

    def transpose_matrix(self):
        result = SparseMatrix(self.cols, self.rows)

        for (row, col), value in self.elements.items():
            result.set_element(col, row, value)

        return result

matrix1 = SparseMatrix.input_matrix()
matrix2 = SparseMatrix.input_matrix()

print("\nМатрица 1:")
matrix1.print_matrix()

print("\nМатрица 2:")
matrix2.print_matrix()

sum_matrix = matrix1.add_matrices(matrix2)
print("\nСумма матриц:")
sum_matrix.print_matrix()

product_matrix = matrix1.multiply_matrices(matrix2)
print("\nПроизведение матриц:")
product_matrix.print_matrix()

transpose_matrix1 = matrix1.transpose_matrix()
print("\nТранспонированная матрица 1:")
transpose_matrix1.print_matrix()


# Задание 4 
class SubjectIndex:
    def __init__(self):
        self.index = {}

    def add_entry(self, word, page_numbers):
        if word in self.index:
            self.index[word].extend(page_numbers)
        else:
            self.index[word] = page_numbers

    def remove_entry(self, word):
        if word in self.index:
            del self.index[word]

    def get_page_numbers(self, word):
        return self.index.get(word, [])

    def print_index(self):
        for word, page_numbers in self.index.items():
            print(f"{word}: {', '.join(map(str, page_numbers))}")

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.index, file)

    def load_from_file(self, filename):
        with open(filename, 'r') as file:
            self.index = json.load(file)


subject_index = SubjectIndex()

subject_index.add_entry("Python", [1, 5, 10])
subject_index.add_entry("Programming", [3, 8, 12])
subject_index.add_entry("Data Science", [6, 9, 15])

print("Предметный указатель:")
subject_index.print_index()

subject_index.save_to_file("subject_index.json")

subject_index.load_from_file("subject_index.json")

search_word = "Python"
print(f"\nНомера страниц для слова '{search_word}': {subject_index.get_page_numbers(search_word)}")

word_to_remove = "Programming"
subject_index.remove_entry(word_to_remove)

print("\nОбновленный предметный указатель:")
subject_index.print_index()



# Задане 5
class AddressBook:
    def __init__(self):
        self.records = []

    def add_record(self, name, birthdate, phone_numbers):
        record = {
            'name': name,
            'birthdate': birthdate,
            'phone_numbers': phone_numbers
        }
        self.records.append(record)

    def remove_record(self, name):
        self.records = [record for record in self.records if record['name'] != name]

    def search_records(self, key, value):
        return [record for record in self.records if record.get(key) == value]

    def print_address_book(self):
        for record in self.records:
            print(f"Имя: {record['name']}, Дата рождения: {record['birthdate']}, Телефоны: {', '.join(record['phone_numbers'])}")

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.records, file)

    def load_from_file(self, filename):
        with open(filename, 'r') as file:
            self.records = json.load(file)



address_book = AddressBook()

# Добавление записей в книжку
address_book.add_record("Иванов", "01.01.1990", ["123456", "789012"])
address_book.add_record("Петров", "15.05.1985", ["111222", "333444"])
address_book.add_record("Сидоров", "30.09.1995", ["555666", "777888"])

# Вывод адресной книжки на экран
print("Адресная книжка:")
address_book.print_address_book()

# Сохранение книжки в файл
address_book.save_to_file("address_book.json")

# Загрузка книжки из файла
address_book.load_from_file("address_book.json")

# Вывод записей с заданным признаком (по имени, дате рождения или номеру телефона)
search_key = "Иванов"
search_result = address_book.search_records("name", search_key)
print(f"\nЗаписи с признаком '{search_key}':")
address_book.print_address_book()

# Удаление записи из книжки
record_to_remove = "Петров"
address_book.remove_record(record_to_remove)

# Вывод обновленной книжки
print("\nОбновленная адресная книжка:")
address_book.print_address_book()





# Задание 6
class Book:
    def __init__(self, title, author, total_copies, on_hand_copies):
        self.title = title
        self.author = author
        self.total_copies = total_copies
        self.on_hand_copies = on_hand_copies

class LibraryCatalog:
    def __init__(self):
        self.books = []

    def add_book(self, title, author, total_copies, on_hand_copies):
        book = Book(title, author, total_copies, on_hand_copies)
        self.books.append(book)

    def remove_book(self, title):
        self.books = [book for book in self.books if book.title != title]

    def search_books(self, key, value):
        return [book for book in self.books if getattr(book, key) == value]

    def print_catalog(self):
        for book in self.books:
            print(f"Название: {book.title}, Автор: {book.author}, Всего экземпляров: {book.total_copies}, Экземпляров на руках: {book.on_hand_copies}")

    def borrow_book(self, title):
        for book in self.books:
            if book.title == title and book.on_hand_copies > 0:
                book.on_hand_copies -= 1
                print(f"Книга '{title}' успешно взята на руки.")
                return
        print(f"Книга '{title}' не может быть взята на руки.")

    def return_book(self, title):
        for book in self.books:
            if book.title == title and book.on_hand_copies < book.total_copies:
                book.on_hand_copies += 1
                print(f"Книга '{title}' успешно возвращена в библиотеку.")
                return
        print(f"Книга '{title}' не может быть возвращена в библиотеку.")

library_catalog = LibraryCatalog()

library_catalog.add_book("Преступление и наказание", "Федор Достоевский", 5, 3)
library_catalog.add_book("1984", "Джордж Оруэлл", 8, 5)
library_catalog.add_book("Мастер и Маргарита", "Михаил Булгаков", 6, 2)

print("Каталог библиотеки:")
library_catalog.print_catalog()

# Сохранение каталога в файл
with open("library_catalog.json", 'w') as file:
    json.dump([vars(book) for book in library_catalog.books], file)

# Удаление книги из каталога
book_to_remove = "1984"
library_catalog.remove_book(book_to_remove)

# Вывод обновленного каталога
print("\nОбновленный каталог библиотеки:")
library_catalog.print_catalog()

# Загрузка каталога из файла
with open("library_catalog.json", 'r') as file:
    data = json.load(file)
    library_catalog.books = [Book(**book_data) for book_data in data]

# Взятие книги на руки
book_to_borrow = "Преступление и наказание"
library_catalog.borrow_book(book_to_borrow)

# Возвращение книги в библиотеку
book_to_return = "Преступление и наказание"
library_catalog.return_book(book_to_return)

# Вывод обновленного каталога
print("\nОбновленный каталог библиотеки:")
library_catalog.print_catalog()




# Задание 7
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def print_even_odd_reverse(node):
    stack_even = [] 
    queue_odd = [] 

    while node:
        if node.data % 2 == 0:
            stack_even.append(node.data)
        else:
            queue_odd.append(node.data)
        node = node.next

    print("Четные числа в обратном порядке:")
    while stack_even:
        print(stack_even.pop(), end=" ")

    print("\nНечетные числа в прямом порядке:")
    for num in queue_odd:
        print(num, end=" ")

# Создаю пример односвязного списка: 1 -> 2 -> 3 -> 4 -> 5
head = Node(1)
head.next = Node(2)
head.next.next = Node(3)
head.next.next.next = Node(4)
head.next.next.next.next = Node(5)

print_even_odd_reverse(head)




# Задание 8
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def print_negative_positive(node):
    queue_negative = [] 
    queue_positive = [] 

    while node:
        if node.data < 0:
            queue_negative.append(node.data)
        elif node.data > 0:
            queue_positive.append(node.data)
        node = node.next

    print("Отрицательные числа:")
    while queue_negative:
        print(queue_negative.pop(0), end=" ")

    print("\nПоложительные числа:")
    while queue_positive:
        print(queue_positive.pop(0), end=" ")

# Создаю пример односвязного списка: -3 -> 2 -> 0 -> -5 -> 4
head = Node(-3)
head.next = Node(2)
head.next.next = Node(0)
head.next.next.next = Node(-5)
head.next.next.next.next = Node(4)

print_negative_positive(head)




# Задание 9
def decimal_to_binary(decimal_number):
    stack = []

    while decimal_number > 0:
        remainder = decimal_number % 2
        stack.append(remainder)
        decimal_number //= 2

    binary_representation = ""
    while stack:
        binary_representation += str(stack.pop())

    return binary_representation

decimal_number = 19
binary_representation = decimal_to_binary(decimal_number)

print(f"Двоичное представление числа {decimal_number}: {binary_representation}")




# Задание 10
def print_exit_path(maze, start_row, start_col):
    rows = len(maze)
    cols = len(maze[0])
    stack = []

    def is_valid_move(row, col):
        return 0 <= row < rows and 0 <= col < cols and maze[row][col] == 0

    def is_exit(row, col):
        return row == 0 or row == rows - 1 or col == 0 or col == cols - 1

    def print_path():
        while stack:
            row, col = stack.pop()
            print(f"({row}, {col}) -> ", end="")

    stack.append((start_row, start_col))
    
    while stack:
        row, col = stack[-1]
        maze[row][col] = 2  # Помечаю текущую позицию путника

        if is_exit(row, col):
            print("Путник вышел из лабиринта. Путь:")
            print_path()
            return

        if is_valid_move(row - 1, col):  # Вверх
            stack.append((row - 1, col))
        elif is_valid_move(row + 1, col):  # Вниз
            stack.append((row + 1, col))
        elif is_valid_move(row, col - 1):  # Влево
            stack.append((row, col - 1))
        elif is_valid_move(row, col + 1):  # Вправо
            stack.append((row, col + 1))
        else:
            stack.pop()  # Если нет доступных направлений, возвращаюсь назад

    print("Путник не смог выйти из лабиринта.")

maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

print_exit_path(maze, 0, 2)





# Задание 11
def print_bracket_positions(expression):
    stack = []
    result = []

    for i, char in enumerate(expression):
        if char == '(':
            stack.append(i + 1)  # Сохраняю позицию открывающей скобки
        elif char == ')':
            if stack:
                opening_position = stack.pop()
                closing_position = i + 1
                result.append((opening_position, closing_position))

    # Сортирую по возрастанию номеров открывающих скобок
    result.sort(key=lambda x: x[0])

    print("Пары скобок, упорядоченные по открывающим скобкам:")
    for opening, closing in result:
        print(f"({opening}, {closing})")

    # Сортирую по возрастанию номеров закрывающих скобок
    result.sort(key=lambda x: x[1])

    print("\nПары скобок, упорядоченные по закрывающим скобкам:")
    for opening, closing in result:
        print(f"({opening}, {closing})")

expression = "(a + (b * c) - (d / e))"
print_bracket_positions(expression)


# Задание 12
def evaluate_expression(expression):
    index = [0]  # Использую список, чтобы хранить индекс и передавать его по ссылке

    def get_digit():
        start = index[0]
        while index[0] < len(expression) and expression[index[0]].isdigit():
            index[0] += 1
        return int(expression[start:index[0]])

    def evaluate_formula():
        char = expression[index[0]]
        if char.isdigit():
            return get_digit()
        elif char == 'M':
            index[0] += 1  # Пропускаю 'M'
            index[0] += 1  # Пропускаю '('
            left = evaluate_formula()
            index[0] += 1  # Пропускаю ','
            right = evaluate_formula()
            index[0] += 1  # Пропускаю ')'
            return max(left, right)
        elif char == 'm':
            index[0] += 1  # Пропускаю 'm'
            index[0] += 1  # Пропускаю '('
            left = evaluate_formula()
            index[0] += 1  # Пропускаю ','
            right = evaluate_formula()
            index[0] += 1  # Пропускаю ')'
            return min(left, right)
        else:
            raise ValueError(f"Неизвестный символ: {char}")

    return evaluate_formula()

math_expression = "M(3,m(5,8))"
result = evaluate_expression(math_expression)
print(f"Результат вычисления выражения {math_expression}: {result}")








# Задание 13
def evaluate_logical_expression(expression):
    index = [0]  # Использую список, чтобы хранить индекс и передавать его по ссылке

    def evaluate_formula():
        char = expression[index[0]]
        if char == 'T':
            index[0] += 1
            return True
        elif char == 'F':
            index[0] += 1
            return False
        elif char == 'A':
            index[0] += 3  # Пропускаю 'And'
            index[0] += 1  # Пропускаю '('
            left = evaluate_formula()
            index[0] += 1  # Пропускаю ','
            right = evaluate_formula()
            index[0] += 1  # Пропускаю ')'
            return left and right
        elif char == 'O':
            index[0] += 2  # Пропускаю 'Or'
            index[0] += 1  # Пропускаю '('
            left = evaluate_formula()
            index[0] += 1  # Пропускаю ','
            right = evaluate_formula()
            index[0] += 1  # Пропускаю ')'
            return left or right
        elif char == 'N':
            index[0] += 2  # Пропускаю 'Not'
            index[0] += 1  # Пропускаю '('
            operand = evaluate_formula()
            index[0] += 1  # Пропускаю ')'
            return not operand
        else:
            raise ValueError(f"Неизвестный символ: {char}")

    return evaluate_formula()

logical_expression = "Or(And(T, F), Not(F))"
result = evaluate_logical_expression(logical_expression)
print(f"Результат вычисления выражения {logical_expression}: {result}")






# Задание 14
def evaluate_postfix(expression):
    stack = []

    for char in expression:
        if char.isdigit():
            stack.append(int(char))
        else:
            operand2 = stack.pop()
            operand1 = stack.pop()

            if char == '+':
                result = operand1 + operand2
            elif char == '-':
                result = operand1 - operand2
            elif char == '*':
                result = operand1 * operand2
            elif char == '/':
                result = operand1 / operand2 
            else:
                raise ValueError(f"Неизвестный оператор: {char}")

            stack.append(result)

    return stack.pop()

postfix_expression = "34+2*"
result = evaluate_postfix(postfix_expression)
print(f"Результат вычисления постфиксного выражения {postfix_expression}: {result}")




# Задание 15
def evaluate_postfix(expression):
    stack = []

    for char in expression:
        if char.isdigit():
            stack.append(int(char))
        elif char in {'+', '-', '*', '/'}:
            operand2 = stack.pop()
            operand1 = stack.pop()

            if char == '+':
                result = operand1 + operand2
            elif char == '-':
                result = operand1 - operand2
            elif char == '*':
                result = operand1 * operand2
            elif char == '/':
                result = operand1 / operand2 
            else:
                raise ValueError(f"Неизвестный оператор: {char}")

            stack.append(result)

    return stack.pop()

postfix_expression = "34+2*"
result = evaluate_postfix(postfix_expression)
print(f"Результат вычисления постфиксного выражения {postfix_expression}: {result}")






