# neuro_symbolic_solver.py
# ИСПРАВЛЕННАЯ версия - корректная обработка вложенных кванторов и сложных формул

import re
from typing import List, Tuple, Optional, Dict, Set


# ====================== AST ======================
class Formula: 
    def free_vars(self) -> Set[str]:
        return set()


class Pred(Formula):
    def __init__(self, name: str, args: List[str]):
        self.name, self.args = name, args

    def __str__(self): 
        if self.args:
            return f"{self.name}({','.join(self.args)})"
        return self.name

    def __repr__(self): return str(self)
    
    def free_vars(self) -> Set[str]:
        return {arg for arg in self.args if arg.islower()}


class Not(Formula):
    def __init__(self, f: Formula): self.f = f

    def __str__(self): 
        if isinstance(self.f, (And, Or, Implies, ForAll, Exists)):
            return f"¬({self.f})"
        return f"¬{self.f}"
    
    def free_vars(self) -> Set[str]:
        return self.f.free_vars()


class And(Formula):
    def __init__(self, left: Formula, right: Formula): 
        self.left, self.right = left, right

    def __str__(self): 
        return f"({self.left} ∧ {self.right})"
    
    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()


class Or(Formula):
    def __init__(self, left: Formula, right: Formula): 
        self.left, self.right = left, right

    def __str__(self): 
        return f"({self.left} ∨ {self.right})"
    
    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()


class Implies(Formula):
    def __init__(self, left: Formula, right: Formula): 
        self.left, self.right = left, right

    def __str__(self): 
        return f"({self.left} → {self.right})"
    
    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()


class ForAll(Formula):
    def __init__(self, var: str, f: Formula): 
        self.var, self.f = var, f

    def __str__(self): 
        return f"∀{self.var}({self.f})"
    
    def free_vars(self) -> Set[str]:
        return self.f.free_vars() - {self.var}


class Exists(Formula):
    def __init__(self, var: str, f: Formula): 
        self.var, self.f = var, f

    def __str__(self): 
        return f"∃{self.var}({self.f})"
    
    def free_vars(self) -> Set[str]:
        return self.f.free_vars() - {self.var}


# ====================== УТИЛИТА: split_top_level ======================
def split_top_level(text: str, sep: str = ',') -> List[str]:
    """
    Split text on sep but only on top-level (not inside parentheses).
    Returns list of trimmed parts.
    """
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
        elif ch == sep and depth == 0:
            part = ''.join(cur).strip()
            if part:
                parts.append(part)
            cur = []
        else:
            cur.append(ch)
        i += 1
    last = ''.join(cur).strip()
    if last:
        parts.append(last)
    return parts


# ====================== ПАРСЕР ДЛЯ ЛЮБЫХ ФОРМУЛ ======================
def parse_formula(s: str) -> Formula:
    s = s.strip()
    
    # Удаляем внешние скобки если есть сбалансированно
    while (s.startswith('(') and s.endswith(')') and 
           is_balanced(s[1:-1])):
        s = s[1:-1].strip()
    
    # Кванторы - исправлено для вложенных формул
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


def parse_predicate(s: str) -> Pred:
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


# ====================== ПРЕОБРАЗОВАНИЕ В КНФ И СКОЛЕМИЗАЦИЯ ======================
class Skolemizer:
    def __init__(self):
        self.skolem_counter = 0
        self.bound_vars_stack = []  # Стек для отслеживания связанных переменных
    
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
        if bound_vars is None:
            bound_vars = []
        
        if isinstance(formula, ForAll):
            # ∀x P(x) -> продолжаем с новыми связанными переменными
            new_bound_vars = bound_vars + [formula.var]
            return ForAll(formula.var, self.skolemize(formula.f, new_bound_vars))
        
        elif isinstance(formula, Exists):
            # ∃x P(x) -> P(sk) где sk - новая сколемовская функция от всех внешних связанных переменных
            skolem_term = self.new_skolem_func(bound_vars, formula.var)
            return self.substitute(formula.f, formula.var, skolem_term)
        
        elif isinstance(formula, And):
            return And(self.skolemize(formula.left, bound_vars), 
                      self.skolemize(formula.right, bound_vars))
        
        elif isinstance(formula, Or):
            return Or(self.skolemize(formula.left, bound_vars), 
                     self.skolemize(formula.right, bound_vars))
        
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
    
    def substitute(self, formula: Formula, var: str, replacement: str) -> Formula:
        """Подставляет терм вместо переменной в формуле"""
        if isinstance(formula, Pred):
            new_args = [replacement if arg == var else arg for arg in formula.args]
            return Pred(formula.name, new_args)
        
        elif isinstance(formula, Not):
            return Not(self.substitute(formula.f, var, replacement))
        
        elif isinstance(formula, And):
            return And(self.substitute(formula.left, var, replacement),
                      self.substitute(formula.right, var, replacement))
        
        elif isinstance(formula, Or):
            return Or(self.substitute(formula.left, var, replacement),
                     self.substitute(formula.right, var, replacement))
        
        elif isinstance(formula, Implies):
            return Implies(self.substitute(formula.left, var, replacement),
                          self.substitute(formula.right, var, replacement))
        
        elif isinstance(formula, ForAll):
            if formula.var == var:
                return formula  # Не подставляем в связанную переменную
            else:
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
        
        else:
            return Or(left, right)
    
    elif isinstance(formula, And):
        return And(distribute_or_over_and(formula.left),
                  distribute_or_over_and(formula.right))
    
    elif isinstance(formula, Not):
        return Not(distribute_or_over_and(formula.f))
    
    elif isinstance(formula, (ForAll, Exists)):
        # Для кванторов - рекурсивно применяем к внутренней формуле
        return type(formula)(formula.var, distribute_or_over_and(formula.f))
    
    else:
        return formula


def remove_quantifiers(formula: Formula) -> Formula:
    """Удаляет кванторы всеобщности после сколемизации"""
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


def cnf_to_string(cnf: Formula) -> str:
    """Преобразует КНФ в строку с клаузами через запятую"""
    clauses = cnf_to_clauses(cnf)
    clause_strings = []
    
    for clause in clauses:
        if isinstance(clause, Or):
            # Собираем все литералы в дизъюнкте
            literals = []
            stack = [clause]
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


# ====================== ResolutionEngine ======================
class Literal:
    def __init__(self, name: str, args: List[str], negated: bool = False):
        self.name = name
        self.args = tuple(args)
        self.negated = negated

    def negate(self): 
        return Literal(self.name, list(self.args), not self.negated)

    def __repr__(self):
        sign = "¬" if self.negated else ""
        args = ','.join(self.args)
        return f"{sign}{self.name}({args})" if args else f"{sign}{self.name}"

    def __eq__(self, other): 
        return isinstance(other, Literal) and (self.name, self.args, self.negated) == (
        other.name, other.args, other.negated)

    def __hash__(self): 
        return hash((self.name, self.args, self.negated))


def parse_cnf(text: str) -> List[List[Literal]]:
    """
    Разбиваем на клаузы по верхнеуровневым запятым
    """
    clauses: List[List[Literal]] = []
    for clause_str in split_top_level(text, sep=','):
        clause_str = clause_str.strip()
        if not clause_str:
            continue
        lits: List[Literal] = []
        for part in re.split(r'\s*∨\s*', clause_str):
            part = part.strip()
            if not part:
                continue
            negated = part.startswith('¬')
            if negated:
                part = part[1:].strip()
            if '(' in part and part.endswith(')'):
                name, args_part = part.split('(', 1)
                args = [a.strip() for a in args_part.rstrip(')').split(',')]
            else:
                name, args = part, []
            lits.append(Literal(name, args, negated))
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


def apply_subst_to_clause(clause: List[Literal], subst: Dict[str, str]) -> List[Literal]:
    return [apply_subst_to_literal(lit, subst) for lit in clause]


class ResolutionEngine:
    def prove(self, cnf_string: str) -> Tuple[bool, List[str]]:
        clauses = parse_cnf(cnf_string)
        steps: List[str] = []
        used_pairs = set()
        step = 1

        def clause_to_str(clause): 
            return " ∨ ".join(str(l) for l in clause) if clause else "Пусто"

        # Основной цикл резолюции
        i = 0
        while i < len(clauses):
            clause1 = clauses[i]
            
            # Проверяем на пустую клаузу
            if not clause1:
                steps.append("Найдена пустой дизъюнкт → противоречие!")
                return True, steps

            j = 0
            while j < len(clauses):
                if i == j:
                    j += 1
                    continue
                    
                clause2 = clauses[j]
                pair_id = (min(i, j), max(i, j))
                
                if pair_id not in used_pairs:
                    used_pairs.add(pair_id)
                    
                    # Пробуем все возможные резолюции
                    for idx1, lit1 in enumerate(clause1):
                        for idx2, lit2 in enumerate(clause2):
                            subst = unify_literals(lit1, lit2)
                            if subst is not None:
                                # Создаем новую клаузу
                                new_clause = []
                                # Добавляем все литералы кроме разрешаемых
                                for k, lit in enumerate(clause1):
                                    if k != idx1:
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
                                for lit in new_clause:
                                    lit_str = str(lit)
                                    if lit_str not in seen:
                                        seen.add(lit_str)
                                        unique_clause.append(lit)
                                
                                expl = f"Шаг {step}: Резолюция ({clause_to_str(clause1)}) и ({clause_to_str(clause2)})"
                                expl += f"\n  Унификация: {lit1} + {lit2} → {subst}"
                                expl += f"\n  Результат: ({clause_to_str(unique_clause)})"
                                steps.append(expl)
                                step += 1
                                
                                if not unique_clause:
                                    steps.append("Получен пустой дизъюнкт → доказательство завершено!")
                                    return True, steps
                                
                                # Проверяем, нет ли уже такой клаузы
                                clause_exists = False
                                for existing_clause in clauses:
                                    if len(existing_clause) == len(unique_clause):
                                        all_match = True
                                        for lit1, lit2 in zip(sorted(existing_clause, key=str), 
                                                             sorted(unique_clause, key=str)):
                                            if lit1 != lit2:
                                                all_match = False
                                                break
                                        if all_match:
                                            clause_exists = True
                                            break
                                
                                if not clause_exists:
                                    clauses.append(unique_clause)
                
                j += 1
            i += 1

        return False, steps + ["Противоречие не найдено"]


# ====================== Главная функция для GUI ======================
def process_task(formula_input: str):
    """Обрабатывает любую формулу: преобразует в КНФ, сколемизирует, доказывает"""
    try:
        # Парсим входную формулу
        formula = parse_formula(formula_input)
        print(f"Парсинг: {formula}")
        
        # Преобразуем в КНФ со сколемизацией
        cnf_formula = to_cnf(formula)
        print(f"КНФ: {cnf_formula}")
        
        # Конвертируем в строку клауз
        cnf_str = cnf_to_string(cnf_formula)
        print(f"Строка КНФ: {cnf_str}")
        
        # Запускаем резолюцию
        engine = ResolutionEngine()
        proven, steps = engine.prove(cnf_str)
        
        return proven, steps, cnf_str
    except Exception as e:
        return False, [f"Ошибка обработки: {str(e)}"], ""
