
# ============================================================
# EMPLOYEE MANAGEMENT SYSTEM USING PYTHON AND MYSQL
# ============================================================


# 1. Import Required Libraries

import mysql.connector

from mysql.connector import Error

from decimal import Decimal, InvalidOperation

# ============================================================
# 2. Database Configuration
# ============================================================

DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "MySQL@2024!Secure"  # Replace with your MySQL password
DB_NAME = "emp"


# ============================================================
# 3. Create Database and Establish Connection
# ============================================================

try:

    # Connect to MySQL Server

    server_connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD
    )

    server_cursor = server_connection.cursor()

    # Create database if it does not exist

    server_cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"
    )

    print("Database created or already exists.")

    server_cursor.close()
    server_connection.close()

    # Connect to the employee database

    con = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

    cursor = con.cursor()

    print("Database connected successfully!")


except Error as err:

    print("\nDatabase Connection Error:", err)

    raise SystemExit


# ============================================================
# 4. Create Employees Table
# ============================================================

try:

    create_table_query = """
    CREATE TABLE IF NOT EXISTS employees (
        id VARCHAR(20) PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        position VARCHAR(100) NOT NULL,
        salary DECIMAL(12, 2) NOT NULL
    )
    """

    cursor.execute(create_table_query)

    con.commit()

    print("Employees table created or already exists.")


except Error as err:

    print("Table Creation Error:", err)

    con.rollback()


# ============================================================
# 5. Function to Check if an Employee Exists
# ============================================================

def check_employee(employee_id):

    sql = "SELECT id FROM employees WHERE id = %s"

    cursor.execute(sql, (employee_id,))

    return cursor.fetchone() is not None


# ============================================================
# 6. Function to Add an Employee
# ============================================================

def add_employee():

    print("\n========== ADD EMPLOYEE ==========")

    employee_id = input("Enter Employee ID: ").strip()

    if not employee_id:

        print("Employee ID cannot be empty.")

        return

    if check_employee(employee_id):

        print("Employee already exists. Please try again.")

        return

    name = input("Enter Employee Name: ").strip()

    if not name:

        print("Employee name cannot be empty.")

        return

    position = input("Enter Employee Position: ").strip()

    if not position:

        print("Employee position cannot be empty.")

        return

    try:

        salary = Decimal(input("Enter Employee Salary: ").strip())

        if not salary.is_finite() or salary < 0:

            print("Please enter a valid non-negative salary.")

            return

    except InvalidOperation:

        print("Invalid salary. Please enter a numeric value.")

        return

    sql = """
    INSERT INTO employees (id, name, position, salary)
    VALUES (%s, %s, %s, %s)
    """

    data = (employee_id, name, position, salary)

    try:

        cursor.execute(sql, data)

        con.commit()

        print("\nEmployee Added Successfully!")

    except Error as err:

        print("Error:", err)

        con.rollback()


# ============================================================
# 7. Function to Remove an Employee
# ============================================================

def remove_employee():

    print("\n========== REMOVE EMPLOYEE ==========")

    employee_id = input("Enter Employee ID: ").strip()

    if not check_employee(employee_id):

        print("Employee does not exist. Please try again.")

        return

    sql = "DELETE FROM employees WHERE id = %s"

    try:

        cursor.execute(sql, (employee_id,))

        con.commit()

        print("\nEmployee Removed Successfully!")

    except Error as err:

        print("Error:", err)

        con.rollback()


# ============================================================
# 8. Function to Promote an Employee
# ============================================================

def promote_employee():

    print("\n========== PROMOTE EMPLOYEE ==========")

    employee_id = input("Enter Employee ID: ").strip()

    if not check_employee(employee_id):

        print("Employee does not exist. Please try again.")

        return

    try:

        amount = Decimal(
            input("Enter Increase in Salary: ").strip()
        )

        if not amount.is_finite() or amount <= 0:

            print("Salary increase must be greater than zero.")

            return

        # Get current salary

        sql_select = """
        SELECT salary FROM employees WHERE id = %s
        """

        cursor.execute(sql_select, (employee_id,))

        result = cursor.fetchone()

        current_salary = result[0]

        # Calculate new salary

        new_salary = current_salary + amount

        # Update employee salary

        sql_update = """
        UPDATE employees
        SET salary = %s
        WHERE id = %s
        """

        cursor.execute(
            sql_update,
            (new_salary, employee_id)
        )

        con.commit()

        print("\nEmployee Promoted Successfully!")

        print("Previous Salary:", current_salary)

        print("Salary Increase:", amount)

        print("New Salary:", new_salary)

    except InvalidOperation:

        print("Invalid salary amount. Please enter a number.")

    except Error as err:

        print("Error:", err)

        con.rollback()


# ============================================================
# 9. Function to Display All Employees
# ============================================================

def display_employees():

    print("\n========== EMPLOYEE RECORDS ==========")

    try:

        sql = "SELECT * FROM employees ORDER BY id"

        cursor.execute(sql)

        employees = cursor.fetchall()

        if not employees:

            print("No employee records found.")

            return

        for employee in employees:

            print("\nEmployee ID:", employee[0])

            print("Employee Name:", employee[1])

            print("Employee Position:", employee[2])

            print("Employee Salary:", employee[3])

            print("------------------------------------")

        print("\nTotal Employees:", len(employees))

    except Error as err:

        print("Error:", err)


# ============================================================
# 10. Function to Display the Menu
# ============================================================

def menu():

    while True:

        print("\n========================================")

        print("     EMPLOYEE MANAGEMENT SYSTEM")

        print("========================================")

        print("1. Add Employee")

        print("2. Remove Employee")

        print("3. Promote Employee")

        print("4. Display Employees")

        print("5. Exit")

        print("========================================")

        choice = input("Enter Your Choice: ").strip()

        if choice == "1":

            add_employee()

        elif choice == "2":

            remove_employee()

        elif choice == "3":

            promote_employee()

        elif choice == "4":

            display_employees()

        elif choice == "5":

            print("\nExiting the program. Goodbye!")

            break

        else:

            print("\nInvalid Choice! Please try again.")


# ============================================================
# 11. Main Program
# ============================================================

if __name__ == "__main__":

    try:

        menu()

    except Error as err:

        print("\nDatabase Error:", err)

    finally:

        # Close database resources

        if cursor:

            cursor.close()

        if con and con.is_connected():

            con.close()

        print("\nDatabase Connection Closed.")


print("Done")
