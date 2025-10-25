"""Simple command-line calculator.

This module provides basic arithmetic operations and a CLI interface that lets
users perform calculations directly from the command line. Supported
operations include addition, subtraction, multiplication, division,
exponentiation, and factorial.

Usage examples::

    $ python calculator.py add 1 2 3
    6.0

    $ python calculator.py multiply 2 3
    6.0

    $ python calculator.py divide 10 2
    5.0

    $ python calculator.py pow 2 10
    1024.0

    $ python calculator.py factorial 5
    120.0

"""
from __future__ import annotations

import argparse
import math
from typing import Callable, Dict, Iterable, List

Number = float


def add(numbers: Iterable[Number]) -> Number:
    """Return the sum of the provided numbers."""
    total = 0.0
    for number in numbers:
        total += number
    return total


def subtract(numbers: Iterable[Number]) -> Number:
    """Return the result of subtracting all subsequent numbers from the first."""
    iterator = iter(numbers)
    try:
        result = next(iterator)
    except StopIteration:
        raise ValueError("subtract operation requires at least one operand") from None

    for number in iterator:
        result -= number
    return result


def multiply(numbers: Iterable[Number]) -> Number:
    """Return the product of the provided numbers."""
    iterator = iter(numbers)
    try:
        product = next(iterator)
    except StopIteration:
        raise ValueError("multiply operation requires at least one operand") from None

    for number in iterator:
        product *= number
    return product


def divide(numbers: Iterable[Number]) -> Number:
    """Return the result of dividing the first number by the subsequent numbers."""
    iterator = iter(numbers)
    try:
        result = next(iterator)
    except StopIteration:
        raise ValueError("divide operation requires at least one operand") from None

    for number in iterator:
        if number == 0:
            raise ZeroDivisionError("division by zero is undefined")
        result /= number
    return result


def power(numbers: Iterable[Number]) -> Number:
    """Return the result of raising the first number to the power of the second."""
    numbers_list = list(numbers)
    if len(numbers_list) != 2:
        raise ValueError("pow operation requires exactly two operands")
    base, exponent = numbers_list
    return base ** exponent


def factorial(numbers: Iterable[Number]) -> Number:
    """Return the factorial of the single provided operand."""
    numbers_list = list(numbers)
    if len(numbers_list) != 1:
        raise ValueError("factorial operation requires exactly one operand")

    (operand,) = numbers_list
    if not operand.is_integer():
        raise ValueError("factorial operand must be an integer")

    integer_operand = int(operand)
    if integer_operand < 0:
        raise ValueError("factorial is undefined for negative numbers")

    return float(math.factorial(integer_operand))


OPERATIONS: Dict[str, Callable[[Iterable[Number]], Number]] = {
    "add": add,
    "sub": subtract,
    "subtract": subtract,
    "mul": multiply,
    "multiply": multiply,
    "div": divide,
    "divide": divide,
    "pow": power,
    "power": power,
    "fact": factorial,
    "factorial": factorial,
}


def parse_arguments(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple command-line calculator")
    parser.add_argument(
        "operation",
        choices=sorted(OPERATIONS.keys()),
        help="Operation to perform",
    )
    parser.add_argument(
        "operands",
        nargs="+",
        type=float,
        help="Operands for the operation",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_arguments(argv)
    operation = OPERATIONS[args.operation]
    result = operation(args.operands)
    print(result)


if __name__ == "__main__":
    main()
