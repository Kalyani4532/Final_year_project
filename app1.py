# ============================
# AI MATH + QUANT SOLVER APP
# ============================

import os, re, json, glob
import numpy as np
import sympy as sp
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

# ============================
# FLASK APP
# ============================
app = Flask(__name__, template_folder="templates")
app.secret_key = "secure_key_123"

# ============================
# ---------- HELPERS ----------
# ============================

def clean_expr(s):
    s = str(s)
    s = s.replace("^", "**").replace("²", "**2")
    s = re.sub(r'(?<=\d)(?=[a-zA-Z])', '*', s)
    s = re.sub(r'(?<=\d)(?=\()', '*', s)
    return s


# ============================
# ---------- CLASSIFIER -------
# ============================

def classify_problem(text):
    t = text.lower()

    if "integrate" in t:
        return "integral"
    if "differentiate" in t:
        return "derivative"
    if "system" in t:
        return "system"
    if "^2" in t or "x**2" in t:
        return "quadratic"
    if "x" in t and "=" in t:
        return "linear"
    if any(w in t for w in ["sin", "cos", "tan", "evaluate"]):
        return "trigonometric"

    if "%" in t or "percent" in t:
        return "percentage"
    if "profit" in t or "loss" in t:
        return "profit_loss"
    if "simple interest" in t:
        return "simple_interest"
    if "compound interest" in t:
        return "compound_interest"
    if "time" in t and "work" in t:
        return "time_work"
    if "speed" in t:
        return "speed_distance"
    if "ratio" in t:
        return "ratio"
    if "average" in t:
        return "average"
    if "age" in t:
        return "ages"
    if "probability" in t:
        return "probability"

    return "unknown"


# ============================
# ---------- MATH SOLVERS -----
# ============================

def solve_linear_steps(problem):
    x = sp.symbols('x')
    lhs, rhs = clean_expr(problem).split("=")
    sol = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)), x)
    return f"x = {sol[0]}", sol[0]


def solve_quadratic_steps(problem):
    x = sp.symbols('x')
    lhs, rhs = clean_expr(problem).split("=")
    poly = sp.expand(sp.sympify(lhs) - sp.sympify(rhs))
    roots = sp.solve(poly, x)
    return f"Roots: {roots}", roots


def solve_integral_steps(problem):
    x = sp.symbols('x')
    expr = clean_expr(problem.replace("integrate", "").replace(":", ""))
    res = sp.integrate(sp.sympify(expr), x)
    return f"∫ {expr} dx = {res} + C", f"{res} + C"


def solve_derivative_steps(problem):
    x = sp.symbols('x')
    expr = clean_expr(problem.replace("differentiate", "").replace(":", ""))
    res = sp.diff(sp.sympify(expr), x)
    return f"d/dx ({expr}) = {res}", res


def solve_trig(problem):
    s = problem.replace("°", "*pi/180")
    expr = sp.sympify(s.replace("evaluate", "").replace(":", ""))
    return "Trig evaluation", sp.simplify(expr)


# ============================
# ---------- QUANT SOLVERS ----
# ============================

def extract_nums(text):
    return list(map(float, re.findall(r'\d+\.?\d*', text)))


def solve_percentage(problem):
    nums = extract_nums(problem)
    price, percent = nums[0], nums[1]
    disc = (percent/100)*price
    return (
        f"Discount = {percent}% of {price} = {disc}\nFinal Price = {price-disc}",
        price-disc
    )


def solve_profit_loss(problem):
    cp, spv = extract_nums(problem)[:2]
    if spv > cp:
        return "Profit", ((spv-cp)/cp)*100
    return "Loss", ((cp-spv)/cp)*100


def solve_simple_interest(problem):
    P, R, T = extract_nums(problem)[:3]
    return "SI", (P*R*T)/100


def solve_compound_interest(problem):
    P, R, T = extract_nums(problem)[:3]
    A = P*(1+R/100)**T
    return "CI", A-P


def solve_time_work(problem):
    A, B = extract_nums(problem)[:2]
    return "Time", 1/(1/A+1/B)


def solve_speed_distance(problem):
    d, t = extract_nums(problem)[:2]
    return "Speed", d/t


def solve_ratio(problem):
    a, b = extract_nums(problem)[:2]
    return "Ratio", f"{a}:{b}"


def solve_average(problem):
    nums = extract_nums(problem)
    return "Average", sum(nums)/len(nums)


def solve_ages(problem):
    a, y = extract_nums(problem)[:2]
    return "Age", a+y


def solve_probability(problem):
    f, t = extract_nums(problem)[:2]
    return "Probability", f/t


# ============================
# ---------- MAIN SOLVER ------
# ============================

def solve_math_problem(problem):
    cat = classify_problem(problem)

    if cat == "linear":
        exp, ans = solve_linear_steps(problem)
    elif cat == "quadratic":
        exp, ans = solve_quadratic_steps(problem)
    elif cat == "integral":
        exp, ans = solve_integral_steps(problem)
    elif cat == "derivative":
        exp, ans = solve_derivative_steps(problem)
    elif cat == "trigonometric":
        exp, ans = solve_trig(problem)
    else:
        return {"category": "unknown", "explanation": "Unsupported", "answer": None}

    return {"category": cat, "explanation": exp, "answer": ans}


# ============================
# ---------- ROUTES -----------
# ============================

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        session["user"] = request.form.to_dict()
        return redirect("/math" if session["user"]["mode"]=="math" else "/quant")
    return render_template("login.html")


@app.route("/math")
def math_page():
    if "user" not in session:
        return redirect("/")
    return render_template("index.html", user=session["user"])


@app.route("/quant")
def quant_page():
    if "user" not in session:
        return redirect("/")
    return render_template("quant.html", user=session["user"])


@app.route("/solve", methods=["POST"])
def solve_math():
    if "user" not in session:
        return jsonify({"error":"Unauthorized"}),401
    return jsonify(solve_math_problem(request.json["problem"]))


@app.route("/solve-quant", methods=["POST"])
def solve_quant():
    if "user" not in session:
        return jsonify({"error":"Unauthorized"}),401

    p = request.json["problem"].lower()
    cat = classify_problem(p)

    solvers = {
        "percentage": solve_percentage,
        "profit_loss": solve_profit_loss,
        "simple_interest": solve_simple_interest,
        "compound_interest": solve_compound_interest,
        "time_work": solve_time_work,
        "speed_distance": solve_speed_distance,
        "ratio": solve_ratio,
        "average": solve_average,
        "ages": solve_ages,
        "probability": solve_probability
    }

    if cat in solvers:
        exp, ans = solvers[cat](p)
        return jsonify({"topic":cat,"explanation":exp,"answer":ans})

    return jsonify({"topic":"unknown","explanation":"Unsupported","answer":None})


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ============================
# ---------- RUN --------------
# ============================

if __name__ == "__main__":
    app.run(debug=True)
