import re
from typing import List, Tuple, Optional, Dict, Set

class Formula:
    def free_vars(self) -> Set[str]: # Метод free_vars() возвращает множество свободных переменных формулы
        return set()


class Pred(Formula): # Представляет атомарное высказывание (предикат)
    def __init__(self, name: str, args: List[str]):
        self.name, self.args = name, args               # name - имя предиката ("Человек", "Любит")
                                                        # args - список аргументов ("Сократ", "x","y")
    def __str__(self):
        if self.args:
            return f"{self.name}({','.join(self.args)})"
        return self.name

    def __repr__(self): return str(self)
    #определяем, что строчные буквы - переменные
    def free_vars(self) -> Set[str]: # возвращает переменные (строчные буквы) из аргументов
        return {arg for arg in self.args if arg.islower()}


class Not(Formula): # отрицание формулы
    def __init__(self, f: Formula): self.f = f

    def __str__(self):
        if isinstance(self.f, (And, Or, Implies, ForAll, Exists)):
            return f"¬({self.f})"
        return f"¬{self.f}"

    def free_vars(self) -> Set[str]:
        return self.f.free_vars()


class And(Formula): # Конъюнкция
    def __init__(self, left: Formula, right: Formula):
        self.left, self.right = left, right

    def __str__(self):
        return f"({self.left} ∧ {self.right})"

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()


class Or(Formula): # Дизъюнкция
    def __init__(self, left: Formula, right: Formula):
        self.left, self.right = left, right

    def __str__(self):
        return f"({self.left} ∨ {self.right})"

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()


class Implies(Formula): # Импликация
    def __init__(self, left: Formula, right: Formula):
        self.left, self.right = left, right

    def __str__(self):
        return f"({self.left} → {self.right})"

    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()


class ForAll(Formula): # Квантор "для всех"
    def __init__(self, var: str, f: Formula):
        self.var, self.f = var, f

    def __str__(self):
        return f"∀{self.var}({self.f})"

    #удаляем переменную из списка свободных
    def free_vars(self) -> Set[str]:
        return self.f.free_vars() - {self.var}


class Exists(Formula): # Квантор "существует"
    def __init__(self, var: str, f: Formula):
        self.var, self.f = var, f

    def __str__(self):
        return f"∃{self.var}({self.f})"

    # удаляем переменную из списка свободных
    def free_vars(self) -> Set[str]:
        return self.f.free_vars() - {self.var}


# разделяет строку по разделителю. Если есть незакрытая скобка, то выражение продолжает обрабатываться, иначе добавляется как предикат
# не трогает запятые внутри скобок. 

def split_top_level(text: str, sep: str = ',') -> List[str]:
    parts = []
    cur = []
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '(':
            depth += 1
            cur.append(ch)
        elif ch == ')':
            depth = max(0, depth - 1)
            cur.append(ch)
        # если же встретили запятую и глубина = 0 (то есть мы не внутри скобок и символ-запятая)
        elif ch == sep and depth == 0:
            part = ''.join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch) #добавляем символ
        i += 1 #переход к след
    last = ''.join(cur).strip() #добавляем последний кусок
    if last:
        parts.append(last)
    return parts

def parse_formula(s: str) -> Formula: # преобразует строковое представление логической формулы в виде объектов класса Formula
    s = s.strip() #удаляем пробелы в начале и в конце строки

    # Удаляем внешние скобки
    while (s.startswith('(') and s.endswith(')') and
           #проверяем что открыв и закрыв скобок одинаково
           is_balanced(s[1:-1])):
        s = s[1:-1].strip()

    # Кванторы
    if s.startswith('∀'):
        match = re.match(r'∀(\w+)\s*\((.*)\)', s)
        if match:
            var, rest = match.groups()
            return ForAll(var, parse_formula(rest))
        else:
            # Попробуем без скобок
            match = re.match(r'∀(\w+)\s*(.*)', s)
            if match:
                var, rest = match.groups()
                return ForAll(var, parse_formula(rest))

    if s.startswith('∃'):
        match = re.match(r'∃(\w+)\s*\((.*)\)', s)
        if match:
            var, rest = match.groups()
            return Exists(var, parse_formula(rest))
        else:
            # Попробуем без скобок
            match = re.match(r'∃(\w+)\s*(.*)', s)
            if match:
                var, rest = match.groups()
                return Exists(var, parse_formula(rest))

    # Импликация
    if '→' in s:
        parts = split_top_level(s, '→')
        if len(parts) == 2:
            return Implies(parse_formula(parts[0]), parse_formula(parts[1]))
    # Конъюнкция
    if '∧' in s:
        parts = split_top_level(s, '∧')
        if len(parts) >= 2:
            result = parse_formula(parts[0])
            for part in parts[1:]:
                result = And(result, parse_formula(part))
            return result

    # Дизъюнкция
    if '∨' in s:
        parts = split_top_level(s, '∨')
        if len(parts) >= 2:
            result = parse_formula(parts[0])
            for part in parts[1:]:
                result = Or(result, parse_formula(part))
            return result

    # Отрицание
    if s.startswith('¬'):
        return Not(parse_formula(s[1:].strip()))

    # Предикат
    return parse_predicate(s)


def is_balanced(s: str) -> bool:
    """Проверяет баланс скобок в строке"""
    count = 0
    for char in s:
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
            if count < 0:
                return False
    return count == 0


def parse_predicate(s: str) -> Pred: # преобразует строковое представление предиката в объект класса Pred
    s = s.strip()
    if '(' not in s:
        return Pred(s, [])

    open_pos = s.find('(')
    name = s[:open_pos].strip()
    args_str = s[open_pos + 1: -1]  # всё внутри скобок

    # Разбиваем аргументы по запятым ТОЛЬКО на верхнем уровне
    args = []
    depth = 0
    start = 0
    for i, c in enumerate(args_str + ','):  # добавляем запятую в конец для удобства
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            arg = args_str[start:i].strip()
            if arg:
                args.append(arg)
            start = i + 1

    return Pred(name, args)

class Skolemizer:
    def __init__(self):
        self.skolem_counter = 0
        self.bound_vars_stack = []
        
    #создает сколемовскую переменную
    def new_skolem_const(self, base_name: str = "sk") -> str:
        self.skolem_counter += 1
        return f"{base_name}_{self.skolem_counter}"

    def new_skolem_func(self, bound_vars: List[str], base_name: str = "f") -> str:
        """Создает сколемовскую функцию с аргументами"""
        self.skolem_counter += 1
        if bound_vars:
            return f"{base_name}_{self.skolem_counter}({','.join(bound_vars)})"
        else:
            return f"{base_name}_{self.skolem_counter}"

    def skolemize(self, formula: Formula, bound_vars: List[str] = None) -> Formula:
        if bound_vars is None: #переменные, связанные внешним квантором
            bound_vars = []

        if isinstance(formula, ForAll):
            # ∀x P(x) -> продолжаем с новыми связанными переменными
            #квантор сохраняется, заопминаем, что переменная связана
            new_bound_vars = bound_vars + [formula.var]
            return ForAll(formula.var, self.skolemize(formula.f, new_bound_vars))

        elif isinstance(formula, Exists):
            # ∃x P(x) -> P(sk) где sk - новая сколемовская функция от всех внешних связанных переменных
            #берем все внешние переменные, создаем сколемовскую функцию от них,заменяем на одну переменную
            skolem_term = self.new_skolem_func(bound_vars, formula.var)
            return self.substitute(formula.f, formula.var, skolem_term)

        #рекурсивно применяем сколемизацию к подформулам
        elif isinstance(formula, And):
            return And(self.skolemize(formula.left, bound_vars),
                       self.skolemize(formula.right, bound_vars))

        elif isinstance(formula, Or):
            return Or(self.skolemize(formula.left, bound_vars),
                      self.skolemize(formula.right, bound_vars))

        # в начале расскрываем ->, далее сколемизация
        elif isinstance(formula, Implies):
            # A → B ≡ ¬A ∨ B
            return self.skolemize(Or(Not(formula.left), formula.right), bound_vars)

        elif isinstance(formula, Not):
            inner = formula.f
            if isinstance(inner, ForAll):
                # ¬∀x P(x) ≡ ∃x ¬P(x)
                return self.skolemize(Exists(inner.var, Not(inner.f)), bound_vars)
            elif isinstance(inner, Exists):
                # ¬∃x P(x) ≡ ∀x ¬P(x)
                return self.skolemize(ForAll(inner.var, Not(inner.f)), bound_vars)
            elif isinstance(inner, And):
                # ¬(A ∧ B) ≡ ¬A ∨ ¬B
                return self.skolemize(Or(Not(inner.left), Not(inner.right)), bound_vars)
            elif isinstance(inner, Or):
                # ¬(A ∨ B) ≡ ¬A ∧ ¬B
                return self.skolemize(And(Not(inner.left), Not(inner.right)), bound_vars)
            elif isinstance(inner, Implies):
                # ¬(A → B) ≡ A ∧ ¬B
                return self.skolemize(And(inner.left, Not(inner.right)), bound_vars)
            elif isinstance(inner, Not):
                # ¬¬A ≡ A
                return self.skolemize(inner.f, bound_vars)
            else:
                return Not(self.skolemize(inner, bound_vars))

        else:
            return formula

                        #формула,         переменная которую меняем, строка-терм
    def substitute(self, formula: Formula, var: str, replacement: str) -> Formula:
        """Подставляет терм вместо переменной в формуле"""

        if isinstance(formula, Pred):
            #Заменяем переменную var в списке аргументов, если она встречается
            new_args = [replacement if arg == var else arg for arg in formula.args]
            return Pred(formula.name, new_args)

        elif isinstance(formula, Not):
            # Рекурсивно применяем подстановку к подформуле
            return Not(self.substitute(formula.f, var, replacement))

        elif isinstance(formula, And):
            #Рекурсивно обрабатываем обе подформулы
            return And(self.substitute(formula.left, var, replacement),
                       self.substitute(formula.right, var, replacement))

        elif isinstance(formula, Or):
            #Рекурсивно обрабатываем обе подформулы
            return Or(self.substitute(formula.left, var, replacement),
                      self.substitute(formula.right, var, replacement))

        elif isinstance(formula, Implies):
            #Рекурсивно обрабатываем обе подформулы
            return Implies(self.substitute(formula.left, var, replacement),
                           self.substitute(formula.right, var, replacement))

        elif isinstance(formula, ForAll):
            if formula.var == var:
                return formula  # Не подставляем в связанную переменную
            else:
                #Новый объект ForAll с:
                #Той же самой переменной квантора
                #Преобразованной подформулой, где произведена подстановка терма вместо переменной в свободных вхождениях
                return ForAll(formula.var, self.substitute(formula.f, var, replacement))

        elif isinstance(formula, Exists):
            if formula.var == var:
                return formula  # Не подставляем в связанную переменную
            else:
                return Exists(formula.var, self.substitute(formula.f, var, replacement))

        return formula


def to_nnf(formula: Formula) -> Formula:
    """Приводит формулу к отрицательной нормальной форме (NNF)"""
    if isinstance(formula, Not):
        inner = formula.f
        if isinstance(inner, Not):
            return to_nnf(inner.f)  # ¬¬A → A
        elif isinstance(inner, And):
            # ¬(A ∧ B) → ¬A ∨ ¬B
            return Or(to_nnf(Not(inner.left)), to_nnf(Not(inner.right)))
        elif isinstance(inner, Or):
            # ¬(A ∨ B) → ¬A ∧ ¬B
            return And(to_nnf(Not(inner.left)), to_nnf(Not(inner.right)))
        elif isinstance(inner, Implies):
            # ¬(A → B) → A ∧ ¬B
            return And(to_nnf(inner.left), to_nnf(Not(inner.right)))
        elif isinstance(inner, ForAll):
            # ¬∀x P → ∃x ¬P
            return Exists(inner.var, to_nnf(Not(inner.f)))
        elif isinstance(inner, Exists):
            # ¬∃x P → ∀x ¬P
            return ForAll(inner.var, to_nnf(Not(inner.f)))
        else:
            return Not(to_nnf(inner))

    elif isinstance(formula, And):
        return And(to_nnf(formula.left), to_nnf(formula.right))

    elif isinstance(formula, Or):
        return Or(to_nnf(formula.left), to_nnf(formula.right))

    elif isinstance(formula, Implies):
        # A → B → ¬A ∨ B
        return Or(to_nnf(Not(formula.left)), to_nnf(formula.right))

    elif isinstance(formula, ForAll):
        return ForAll(formula.var, to_nnf(formula.f))

    elif isinstance(formula, Exists):
        return Exists(formula.var, to_nnf(formula.f))

    else:
        return formula


def distribute_or_over_and(formula: Formula) -> Formula:
    """Дистрибутивность ∨ над ∧"""
    if isinstance(formula, Or):
        left = distribute_or_over_and(formula.left)
        right = distribute_or_over_and(formula.right)

        if isinstance(left, And):
            # (A ∧ B) ∨ C → (A ∨ C) ∧ (B ∨ C)
            return And(distribute_or_over_and(Or(left.left, right)),
                       distribute_or_over_and(Or(left.right, right)))

        elif isinstance(right, And):
            # A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)
            return And(distribute_or_over_and(Or(left, right.left)),
                       distribute_or_over_and(Or(left, right.right)))

        #Ни одна часть не конъюнкция
        #Когда обе части уже не содержат ∧ под ∨
        else:
            return Or(left, right)

    #Рекурсивно применяем дистрибутивность к обеим подформулам.
    elif isinstance(formula, And):
        return And(distribute_or_over_and(formula.left),
                   distribute_or_over_and(formula.right))

    elif isinstance(formula, Not):
        return Not(distribute_or_over_and(formula.f))

    elif isinstance(formula, (ForAll, Exists)):
        # Для кванторов - определяем его класс, рекурсивно применяем к внутренней формуле
        return type(formula)(formula.var, distribute_or_over_and(formula.f))

    else:
        return formula


def remove_quantifiers(formula: Formula) -> Formula:
    """Удаляет кванторы всеобщности"""
    if isinstance(formula, ForAll):
        return remove_quantifiers(formula.f)
    elif isinstance(formula, And):
        return And(remove_quantifiers(formula.left), remove_quantifiers(formula.right))
    elif isinstance(formula, Or):
        return Or(remove_quantifiers(formula.left), remove_quantifiers(formula.right))
    elif isinstance(formula, Not):
        return Not(remove_quantifiers(formula.f))
    elif isinstance(formula, Implies):
        return Implies(remove_quantifiers(formula.left), remove_quantifiers(formula.right))
    else:
        return formula


def to_cnf(formula: Formula) -> Formula:
    """Полное преобразование в КНФ"""
    # 1. Приведение к NNF
    nnf_formula = to_nnf(formula)

    # 2. Сколемизация
    skolemizer = Skolemizer()
    skolemized = skolemizer.skolemize(nnf_formula)

    # 3. Удаление кванторов всеобщности
    no_quantifiers = remove_quantifiers(skolemized)

    # 4. Дистрибутивность
    cnf_formula = distribute_or_over_and(no_quantifiers)

    return cnf_formula


def cnf_to_clauses(cnf: Formula) -> List[Formula]:
    """Разбивает КНФ на список дизъюнктов"""
    if isinstance(cnf, And):
        return cnf_to_clauses(cnf.left) + cnf_to_clauses(cnf.right)
    else:
        return [cnf]


    #
    #   КНФ: (A ∨ B) ∧ (C ∨ D) ∧ E
    #   Результат: [A ∨ B, C ∨ D, E]
    #

def cnf_to_string(cnf: Formula) -> str:
    """Преобразует КНФ в строку с клаузами через запятую"""
    clauses = cnf_to_clauses(cnf)
    clause_strings = []

    for clause in clauses:
        if isinstance(clause, Or):
            # Собираем все литералы в дизъюнкте
            literals = []
            stack = [clause]
            #поиск в глубину
            while stack:
                node = stack.pop()
                if isinstance(node, Or):
                    stack.append(node.right)
                    stack.append(node.left)
                elif isinstance(node, Not):
                    if isinstance(node.f, Pred):
                        args_str = ','.join(node.f.args)
                        literals.append(f"¬{node.f.name}({args_str})" if node.f.args else f"¬{node.f.name}")
                    else:
                        literals.append(f"¬{node.f}")
                elif isinstance(node, Pred):
                    args_str = ','.join(node.args)
                    literals.append(f"{node.name}({args_str})" if node.args else node.name)
            clause_strings.append(" ∨ ".join(sorted(literals)))

        elif isinstance(clause, Not):
            if isinstance(clause.f, Pred):
                args_str = ','.join(clause.f.args)
                clause_strings.append(f"¬{clause.f.name}({args_str})" if clause.f.args else f"¬{clause.f.name}")
            else:
                clause_strings.append(f"¬{clause.f}")

        elif isinstance(clause, Pred):
            args_str = ','.join(clause.args)
            clause_strings.append(f"{clause.name}({args_str})" if clause.args else clause.name)

    return ", ".join(clause_strings)

    #
    # Вход: [A ∨ B, C, D ∨ E]
    # Выход: "A ∨ B, C, D ∨ E"
    #


class Literal:
    def __init__(self, name: str, args: List[str], negated: bool = False):
        self.name = name
        self.args = tuple(args)
        self.negated = negated
    #Literal("Любит", ["x", "y"], True)

    #создание отрицания литерала
    def negate(self):
        return Literal(self.name, list(self.args), not self.negated)

    #строковое представление
    def __repr__(self):
        sign = "¬" if self.negated else ""
        args = ','.join(self.args)
        return f"{sign}{self.name}({args})" if args else f"{sign}{self.name}"
    # Literal("Человек", ["Катя"]) -> Выход: Человек(Катя)

    #определяет, когда два литерала считаются одинаковыми
    def __eq__(self, other):
        return isinstance(other, Literal) and (self.name, self.args, self.negated) == (
            other.name, other.args, other.negated)

    def __hash__(self):
        return hash((self.name, self.args, self.negated))
        #хэшируемый кортеж

def parse_cnf(text: str) -> List[List[Literal]]:
    """
    Разбиваем на клаузы по верхнеуровневым запятым
    P(x,y), Q(f(a), b) ∨ R(z), S(t)
    """
    clauses: List[List[Literal]] = []
    #разбиваем по запятым
    for clause_str in split_top_level(text, sep=','):

        #проверка на пустую строку
        clause_str = clause_str.strip()
        if not clause_str:
            continue
        lits: List[Literal] = []

        #разбиение по V с любым количеством пробелов вокруг
        for part in re.split(r'\s*∨\s*', clause_str):
            #удаление пробелов
            part = part.strip()
            if not part:
                continue
            # проверяет, начинается ли литерал с символа отрицания
            negated = part.startswith('¬')
            if negated:
                #удаление отрицания
                part = part[1:].strip()

            # cодержит ли строка символ открывающей скобки (является ли это предикатом с аргументами)
            if '(' in part and part.endswith(')'): # и заканчивается ли строка символом закрывающей скобки

                # разделяем строку на первой встреченной открывающей скобке
                name, args_part = part.split('(', 1)
                # Разбираем строку аргументов: удаляем закрывающую скобку с конца, разбиваем по запятым на отдельные аргументы,
                # и для каждого аргумента убираем лишние пробелы вокруг, создавая чистый список аргументов предиката
                args = [a.strip() for a in args_part.rstrip(')').split(',')]

            #обработка предикатов без аргументов
            else:
                name, args = part, []
            #Собирает все обработанные компоненты в объект Literal
            lits.append(Literal(name, args, negated))
        # добавление в результат
        clauses.append(lits)
    return clauses


def unify_literals(l1: Literal, l2: Literal) -> Optional[Dict[str, str]]:
    if l1.name != l2.name or l1.negated == l2.negated or len(l1.args) != len(l2.args):
        return None
    subst: Dict[str, str] = {}
    for a1, a2 in zip(l1.args, l2.args):
        if a1.islower():
            if a1 in subst and subst[a1] != a2:
                return None
            subst[a1] = a2
        elif a2.islower():
            if a2 in subst and subst[a2] != a1:
                return None
            subst[a2] = a1
        elif a1 != a2:
            return None
    return subst


def apply_subst_to_literal(lit: Literal, subst: Dict[str, str]) -> Literal:
    new_args = [subst.get(a, a) for a in lit.args]
    return Literal(lit.name, new_args, lit.negated)


class Literal:
    def __init__(self, name: str, args: List[str], negated: bool = False):
        self.name = name
        self.args = tuple(args)
        self.negated = negated
    #Literal("Любит", ["x", "y"], True)

    #создание отрицания литерала
    def negate(self):
        return Literal(self.name, list(self.args), not self.negated)

    #строковое представление
    def __repr__(self):
        sign = "¬" if self.negated else ""
        args = ','.join(self.args)
        return f"{sign}{self.name}({args})" if args else f"{sign}{self.name}"
    # Literal("Человек", ["Катя"]) -> Выход: Человек(Катя)

    #определяет, когда два литерала считаются одинаковыми
    def __eq__(self, other):
        return isinstance(other, Literal) and (self.name, self.args, self.negated) == (
            other.name, other.args, other.negated)

    def __hash__(self):
        return hash((self.name, self.args, self.negated))
        #хэшируемый кортеж

def parse_cnf(text: str) -> List[List[Literal]]:
    """
    Разбиваем на клаузы по верхнеуровневым запятым
    P(x,y), Q(f(a), b) ∨ R(z), S(t)
    """
    clauses: List[List[Literal]] = []
    #разбиваем по запятым
    for clause_str in split_top_level(text, sep=','):

        #проверка на пустую строку
        clause_str = clause_str.strip()
        if not clause_str:
            continue
        lits: List[Literal] = []

        #разбиение по V с любым количеством пробелов вокруг
        for part in re.split(r'\s*∨\s*', clause_str):
            #удаление пробелов
            part = part.strip()
            if not part:
                continue
            # проверяет, начинается ли литерал с символа отрицания
            negated = part.startswith('¬')
            if negated:
                #удаление отрицания
                part = part[1:].strip()

            # cодержит ли строка символ открывающей скобки (является ли это предикатом с аргументами)
            if '(' in part and part.endswith(')'): # и заканчивается ли строка символом закрывающей скобки

                # разделяем строку на первой встреченной открывающей скобке
                name, args_part = part.split('(', 1)
                # Разбираем строку аргументов: удаляем закрывающую скобку с конца, разбиваем по запятым на отдельные аргументы,
                # и для каждого аргумента убираем лишние пробелы вокруг, создавая чистый список аргументов предиката
                args = [a.strip() for a in args_part.rstrip(')').split(',')]

            #обработка предикатов без аргументов
            else:
                name, args = part, []
            #Собирает все обработанные компоненты в объект Literal
            lits.append(Literal(name, args, negated))
        # добавление в результат
        clauses.append(lits)
    return clauses


def unify_literals(l1: Literal, l2: Literal) -> Optional[Dict[str, str]]:
    #имена предикатов разные, знаки литералов одинаковые, разное кол-во аргументов
    if l1.name != l2.name or l1.negated == l2.negated or len(l1.args) != len(l2.args):
        #унификация невозможна
        return None
    #подстановка
    subst: Dict[str, str] = {}

    # Обрабатываем попарно соответствующие аргументы из двух литералов
    for a1, a2 in zip(l1.args, l2.args):

        # Если Первый аргумент - переменная
        if a1.islower():
            # Если для этой переменной УЖЕ есть подстановка и она конфликтует с текущим значением
            if a1 in subst and subst[a1] != a2:
                return None  # Обнаружен конфликт подстановок - унификация невозможна
            # Сохраняем подстановку: переменная a1 заменяется на значение a2
            subst[a1] = a2

        # Если Второй аргумент - переменная
        elif a2.islower():
            # Если для этой переменной УЖЕ есть подстановка и она КОНФЛИКТУЕТ с текущим значением
            if a2 in subst and subst[a2] != a1:
                return None  # Обнаружен конфликт подстановок - унификация невозможна
            # Сохраняем подстановку: переменная a2 заменяется на значение a1
            subst[a2] = a1

        # СЛУЧАЙ 3: Оба аргумента - константы (не переменные)
        elif a1 != a2:
            return None  # Константы не совпадают - унификация невозможна

        # Если оба аргумента - одинаковые константы, продолжаем без изменений

    #возвращаем  подстановки
    return subst


def apply_subst_to_literal(lit: Literal, subst: Dict[str, str]) -> Literal:
    new_args = [subst.get(a, a) for a in lit.args]
    return Literal(lit.name, new_args, lit.negated)



class ResolutionEngine:
    def prove(self, cnf_string: str) -> Tuple[bool, List[str]]:
        #преобразует строку КНФ в список дизъюнктов
        clauses = parse_cnf(cnf_string)

        steps: List[str] = [] #шаги док-ва
        used_pairs = set()
        step = 1

        def clause_to_str(clause):
            return " ∨ ".join(str(l) for l in clause) if clause else "Пусто"
            #[¬P(x), Q(f(x))] → ¬P(x) ∨ Q(f(x))

        # Основной цикл резолюции
        i = 0
        #цикл пока не обработаны все дизъюнкты
        while i < len(clauses):
            #Извлечение текущего дизъюнкта
            clause1 = clauses[i]

            # Проверяем на пустой дизъюнкт
            if not clause1:
                steps.append("Найден пустой дизъюнкт → противоречие!")
                return True, steps

            # внутренний цикл по всем дизъюнктам
            j = 0
            while j < len(clauses):
                #проверка, что мы не пытаемся применить резолюцию дизъюнкта с самим собой
                if i == j:
                    #увеличение индекса для перехода к следующему дизъюнкту
                    j += 1
                    #пропуск текущей итерации, переход к следующему дизъюнкту
                    continue

                #Выбор второго дизъюнкта
                clause2 = clauses[j]

                #чтобы дизъюнкты не резольвировались дважды
                pair_id = (min(i, j), max(i, j))
                if pair_id not in used_pairs:
                    used_pairs.add(pair_id)

                    # Пробуем все возможные резолюции
                    #цикл по всем парам литералов
                    for idx1, lit1 in enumerate(clause1):
                        for idx2, lit2 in enumerate(clause2):
                            subst = unify_literals(lit1, lit2)
                            # попытка унификации двух литералов из разных дизъюнктов

                            # если литералы унифицируемы
                            if subst is not None:
                                # создаем новую резольвенту
                                new_clause = []
                                # добавляем все литералы кроме разрешаемых

                                #проход по первому дизъюнкту
                                for k, lit in enumerate(clause1):
                                    #если k не равно индексу литерала, который учавствует в унификации
                                    if k != idx1:
                                        #применение  подстановки к текущему литералу,
                                        #добавление преобразованного литерала в новую резольвенту
                                        new_clause.append(apply_subst_to_literal(lit, subst))
                                for k, lit in enumerate(clause2):
                                    if k != idx2:
                                        new_lit = apply_subst_to_literal(lit, subst)
                                        # Проверяем на дубликаты
                                        if new_lit not in new_clause:
                                            new_clause.append(new_lit)

                                # Убираем дубликаты
                                unique_clause = []
                                seen = set()

                                # цикл по всем литералам новой резольвенты
                                for lit in new_clause:
                                    #строковое представление
                                    lit_str = str(lit)
                                    # проверка, не встречался ли уже этот литерал
                                    if lit_str not in seen:
                                        # добавление строкового представления литерала в множество
                                        seen.add(lit_str)
                                        # добавление уникального литерала в дизъюнкт
                                        unique_clause.append(lit)

                                expl = f"Шаг {step}: Резолюция ({clause_to_str(clause1)}) и ({clause_to_str(clause2)})"
                                expl += f"\n  Унификация: {lit1} + {lit2} → {subst}"
                                expl += f"\n  Результат: ({clause_to_str(unique_clause)})"
                                steps.append(expl)
                                step += 1

                                # проверка на пустой дизъюнкт
                                if not unique_clause:
                                    steps.append("Получен пустой дизъюнкт → доказательство завершено!")
                                    return True, steps

                                # Проверка уникальности нового дизъюнкта
                                clause_exists = False
                                # Цикл по всем существующим дизъюнктам
                                for existing_clause in clauses:
                                    # если длины разные - дизъюнкты гарантированно разные
                                    if len(existing_clause) == len(unique_clause):
                                        # флаг полного совпадения
                                        all_match = True

                                        # Сравнение дизъюнктов с учетом коммутативности дизъюнкции
                                        # Сортировка литералов гарантирует корректное сравнение [A,B] ≡ [B,A]
                                        for lit1, lit2 in zip(sorted(existing_clause, key=str),
                                                              sorted(unique_clause, key=str)):

                                            #проверка, что соответствующие литералы  разные
                                            if lit1 != lit2:
                                                all_match = False
                                                break

                                        # проверка, что все литералы совпали
                                        if all_match:
                                            clause_exists = True
                                            #прерываем цикл, так как дубликат уже найден
                                            break

                                if not clause_exists:
                                    clauses.append(unique_clause)

                j += 1
            i += 1

        return False, steps + ["Противоречие не найдено"]