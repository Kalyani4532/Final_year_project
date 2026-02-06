# app.py - AI Math Solver (extended deterministic explanations)
import os
import glob
import json
import re
import numpy as np
from flask import Flask, render_template, request, jsonify
import sympy as sp

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- CONFIG (file names) ----------
VOCAB_CANDIDATES = [
    os.path.join(BASE_DIR, "seq2seq_vocab.json"),
    os.path.join(BASE_DIR, "seq2seq_vocab (1).json"),
    os.path.join(BASE_DIR, "seq2seq_vocab_v1.json"),
]
ENCODER_H5 = os.path.join(BASE_DIR, "encoder_inf.h5")
DECODER_H5 = os.path.join(BASE_DIR, "decoder_inf.h5")
TRAIN_H5   = os.path.join(BASE_DIR, "math_solver_model.h5")  # fallback if no inference models

# ---------- FLASK APP ----------
app = Flask(__name__, template_folder="templates")

# ---------- LOCATE VOCAB FILE ----------
VOCAB_PATH = None
for p in VOCAB_CANDIDATES:
    if os.path.exists(p):
        VOCAB_PATH = p
        break
if VOCAB_PATH is None:
    found = glob.glob(os.path.join(BASE_DIR, "seq2seq_vocab*.json"))
    if found:
        VOCAB_PATH = found[0]

if VOCAB_PATH is None:
    raise FileNotFoundError("Missing vocab JSON (seq2seq_vocab.json). Place it in the project folder.")

# ---------- LOAD VOCAB (support multiple export formats) ----------
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

if "input_token_index" in vocab_data and "target_token_index" in vocab_data:
    input_token_index = {k: int(v) for k, v in vocab_data["input_token_index"].items()}
    target_token_index = {k: int(v) for k, v in vocab_data["target_token_index"].items()}
    if "reverse_target_char_index" in vocab_data:
        reverse_target_char_index = {int(k): v for k, v in vocab_data["reverse_target_char_index"].items()}
    else:
        reverse_target_char_index = {i: c for c, i in vocab_data["target_token_index"].items()}
    input_chars = vocab_data.get("input_chars", list(input_token_index.keys()))
    target_chars = vocab_data.get("target_chars", list(target_token_index.keys()))
    num_encoder_tokens = int(vocab_data.get("num_encoder_tokens", len(input_chars)))
    num_decoder_tokens = int(vocab_data.get("num_decoder_tokens", len(target_chars)))
    max_encoder_seq_length = int(vocab_data.get("max_encoder_seq_length", 200))
    max_decoder_seq_length = int(vocab_data.get("max_decoder_seq_length", 200))
else:
    input_chars = vocab_data["input_chars"]
    target_chars = vocab_data["target_chars"]
    input_token_index  = {char: i for i, char in enumerate(input_chars)}
    target_token_index = {char: i for i, char in enumerate(target_chars)}
    reverse_target_char_index = {i: char for char, i in target_token_index.items()}
    num_encoder_tokens = len(input_chars)
    num_decoder_tokens = len(target_chars)
    max_encoder_seq_length = int(vocab_data.get("max_encoder_seq_length", max(len(s) for s in input_chars)))
    max_decoder_seq_length = int(vocab_data.get("max_decoder_seq_length", max(len(s) for s in target_chars)))

print(f"[Vocab] encoder tokens={num_encoder_tokens}, decoder tokens={num_decoder_tokens}")
if "<" not in target_token_index:
    print("WARNING: start token '<' missing in target_token_index.")
if ">" not in target_token_index:
    print("WARNING: end token '>' missing in target_token_index.")

# ---------- SMALL CLEAN HELPER ----------
def clean_expr(s: str) -> str:
    s = str(s)

    # Normalize powers
    s = s.replace("^", "**")
    s = s.replace("Â²", "**2").replace("²", "**2")

    # Normalize minus
    s = s.replace("–", "-")

    # Insert * ONLY between number and variable: 2x → 2*x
    s = re.sub(r'(?<=\d)(?=[a-zA-Z])', '*', s)

    # Insert * between number and bracket: 2(x+1) → 2*(x+1)
    s = re.sub(r'(?<=\d)(?=\()', '*', s)

    return s

def convert_degrees(expr: str) -> str:
    # Converts 30° → (30*pi/180)
    return re.sub(r'(\d+)\s*°', r'(\1*pi/180)', expr)


# ---------- LOAD OR RECONSTRUCT MODELS ----------
encoder_model = None
decoder_model = None

def try_load_inference_models():
    global encoder_model, decoder_model
    if os.path.exists(ENCODER_H5) and os.path.exists(DECODER_H5):
        encoder_model = load_model(ENCODER_H5)
        decoder_model = load_model(DECODER_H5)
        print("[Model] Loaded encoder_inf.h5 and decoder_inf.h5")
        return True
    return False

def try_reconstruct_from_train():
    global encoder_model, decoder_model
    if not os.path.exists(TRAIN_H5):
        return False
    print("[Model] Found training model. Attempting to reconstruct inference models from training .h5 ...")
    trained = load_model(TRAIN_H5)
    name_to_layer = {layer.name: layer for layer in trained.layers}
    try:
        enc_layer = name_to_layer.get("encoder_lstm")
        dec_layer = name_to_layer.get("decoder_lstm")
        latent_dim = None
        if enc_layer is not None:
            cfg = getattr(enc_layer, "get_config", lambda: {})()
            latent_dim = cfg.get("units") if isinstance(cfg, dict) else getattr(enc_layer, "units", None)
        if latent_dim is None and dec_layer is not None:
            cfg = getattr(dec_layer, "get_config", lambda: {})()
            latent_dim = cfg.get("units") if isinstance(cfg, dict) else getattr(dec_layer, "units", None)
        if latent_dim is None:
            raise RuntimeError("Could not determine latent_dim from trained model layers.")
        encoder_inputs_infer = Input(shape=(None, num_encoder_tokens), name="encoder_inputs")
        encoder_lstm_infer = LSTM(latent_dim, return_state=True, name="encoder_lstm")
        _, state_h, state_c = encoder_lstm_infer(encoder_inputs_infer)
        encoder_model = Model(encoder_inputs_infer, [state_h, state_c])
        decoder_inputs_infer = Input(shape=(None, num_decoder_tokens), name="decoder_inputs")
        dec_state_h = Input(shape=(latent_dim,), name="dec_state_h")
        dec_state_c = Input(shape=(latent_dim,), name="dec_state_c")
        decoder_states_inputs = [dec_state_h, dec_state_c]
        decoder_lstm_infer = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
        decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm_infer(decoder_inputs_infer, initial_state=decoder_states_inputs)
        decoder_dense_infer = Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
        decoder_outputs_inf = decoder_dense_infer(decoder_outputs_inf)
        decoder_model = Model([decoder_inputs_infer] + decoder_states_inputs, [decoder_outputs_inf, state_h_inf, state_c_inf])
        if "encoder_lstm" in name_to_layer:
            encoder_lstm_infer.set_weights(name_to_layer["encoder_lstm"].get_weights())
        if "decoder_lstm" in name_to_layer:
            decoder_lstm_infer.set_weights(name_to_layer["decoder_lstm"].get_weights())
        if "decoder_dense" in name_to_layer:
            decoder_dense_infer.set_weights(name_to_layer["decoder_dense"].get_weights())
        print("[Model] Reconstructed encoder & decoder from training model.")
        return True
    except Exception as e:
        print("[Model] Failed to reconstruct inference models:", str(e))
        return False

loaded = try_load_inference_models()
if not loaded:
    loaded = try_reconstruct_from_train()
if not loaded:
    print("[Model] WARNING: no seq2seq model loaded. seq2seq explanation fallback will be skipped.")
    encoder_model = None
    decoder_model = None

# ---------- SEQ2SEQ DECODE FUNCTION (fallback) ----------
def decode_sequence(input_text, max_decoder_len=None):
    if encoder_model is None or decoder_model is None:
        return ""
    if max_decoder_len is None:
        max_decoder_len = max_decoder_seq_length
    encoder_input = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    for t, ch in enumerate(input_text):
        if t >= max_encoder_seq_length:
            break
        idx = input_token_index.get(ch)
        if idx is not None and 0 <= idx < num_encoder_tokens:
            encoder_input[0, t, idx] = 1.0
    states_value = encoder_model.predict(encoder_input)
    target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
    if "<" in target_token_index:
        target_seq[0, 0, target_token_index["<"]] = 1.0
    decoded_chars = []
    steps = 0
    SAFE_MAX_STEPS = max(200, max_decoder_len)
    while steps < SAFE_MAX_STEPS:
        steps += 1
        out = decoder_model.predict([target_seq] + states_value)
        if isinstance(out, list) and len(out) >= 3:
            output_tokens = out[0]; h = out[1]; c = out[2]
        else:
            output_tokens = out[0] if isinstance(out, list) else out
            h, c = None, None
        probs = output_tokens[0, -1, :]
        sampled_token_index = int(np.argmax(probs))
        sampled_char = reverse_target_char_index.get(sampled_token_index)
        if sampled_char is None or sampled_char == ">":
            break
        decoded_chars.append(sampled_char)
        target_seq = np.zeros((1, 1, num_decoder_tokens), dtype="float32")
        if 0 <= sampled_token_index < num_decoder_tokens:
            target_seq[0, 0, sampled_token_index] = 1.0
        if h is not None and c is not None:
            states_value = [h, c]
    return "".join(decoded_chars).strip()

# ---------- SYMBOLIC (SymPy) SOLVERS FOR EXACT ANSWERS ----------
def solve_equation_exact(problem_text: str, problem_category: str = ""):
    x, y = sp.symbols('x y')
    s = problem_text.strip()
    s_lower = s.lower()
    if "system" in (problem_category or "").lower() or "solve the system" in s_lower:
        s_core = s
        if ":" in s:
            s_core = s.split(":", 1)[1]
        s_core = clean_expr(s_core)
        eq_strs = [e.strip() for e in re.split(r'[;,]', s_core) if "=" in e]
        if len(eq_strs) < 2:
            return None
        eqs = []
        for es in eq_strs[:2]:
            ls, rs = es.split("=", 1)
            eqs.append(sp.Eq(sp.sympify(ls.strip()), sp.sympify(rs.strip())))
        sol_list = sp.solve(eqs, (x, y), dict=True)
        if not sol_list:
            return None
        sol = sol_list[0]
        sx = sp.nsimplify(sol.get(x)); sy = sp.nsimplify(sol.get(y))
        return f"x = {sx}, y = {sy}"
    s_core = s.replace("solve for x:", "").replace("Solve for x:", "").replace("solve:", "").replace("Solve:", "")
    s_core = clean_expr(s_core)
    if "=" not in s_core:
        return None
    ls, rs = s_core.split("=", 1)
    try:
        eq = sp.Eq(sp.sympify(ls.strip()), sp.sympify(rs.strip()))
        sol = sp.solve(eq, sp.symbols('x'))
    except Exception:
        return None
    if not sol:
        return None
    if len(sol) == 1:
        return f"x = {sp.nsimplify(sol[0])}"
    else:
        return " or ".join([f"x = {sp.nsimplify(r)}" for r in sol])

def solve_derivative_exact(problem_text: str):
    x = sp.symbols('x')
    s = problem_text
    s = re.sub(r'(?i)differentiat(e|ion|e):?', '', s)
    s = s.replace("f(x) =", "").replace("y =", "")
    s = clean_expr(s)
    try:
        expr = sp.sympify(s)
        der  = sp.diff(expr, x)
        return str(sp.simplify(der))
    except Exception:
        return None

def solve_integral_exact(problem_text: str):
    x = sp.symbols('x')

    s = re.sub(r'(?i)integrat(e|ion|e):?', '', problem_text)
    s = s.replace("f(x) =", "").replace("dx", "").strip()

    limit_match = re.search(r'on\s*\[\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\]', s)

    lower = upper = None
    if limit_match:
        lower = sp.sympify(limit_match.group(1))
        upper = sp.sympify(limit_match.group(2))
        s = re.sub(r'on\s*\[.*?\]', '', s)

    s = clean_expr(s)

    try:
        expr = sp.sympify(s)
        if lower is not None and upper is not None:
            return str(sp.integrate(expr, (x, lower, upper)))
        else:
            return str(sp.integrate(expr, x)) + " + C"
    except Exception:
        return None




def get_exact_answer(problem_text: str, problem_category: str):
    cat  = (problem_category or "").lower()
    text = problem_text.lower()
    if "integral" in cat or "integr" in text or "∫" in text:
        return solve_integral_exact(problem_text)
    elif "derivative" in cat or "differentiate" in text or "derivative" in text:
        return solve_derivative_exact(problem_text)
    else:
        return solve_equation_exact(problem_text, problem_category)

# ---------- DETERMINISTIC 2x2 SYSTEM STEP-BY-STEP (ELIMINATION) ----------
def solve_system_steps(problem_text: str) -> str:
    x, y = sp.symbols('x y')
    s = problem_text.strip()
    if ":" in s:
        s = s.split(":", 1)[1]
    s = clean_expr(s)
    eq_strs = [e.strip() for e in re.split(r'[;,]', s) if "=" in e]
    if len(eq_strs) < 2:
        return "Could not parse a 2-equation system. Provide two equations separated by comma."
    eq_strs = eq_strs[:2]
    try:
        eqs = []
        for es in eq_strs:
            ls, rs = es.split("=", 1)
            eqs.append(sp.Eq(sp.sympify(ls.strip()), sp.sympify(rs.strip())))
        A, b = sp.linear_eq_to_matrix(eqs, [x, y])
        a1 = sp.simplify(A[0, 0]); b1 = sp.simplify(A[0, 1]); c1 = sp.simplify(b[0])
        a2 = sp.simplify(A[1, 0]); b2 = sp.simplify(A[1, 1]); c2 = sp.simplify(b[1])
        steps = []
        steps.append("Given the system:")
        steps.append(f"  (1) {sp.pretty(eqs[0])}")
        steps.append(f"  (2) {sp.pretty(eqs[1])}")
        steps.append("")
        det = sp.simplify(a1*b2 - a2*b1)
        if det == 0:
            steps.append("The coefficient determinant is 0 (system may be dependent or inconsistent).")
            return "\n".join(steps)
        steps.append("Elimination (eliminate x):")
        steps.append(f"  Multiply (1) by {a2} -> {sp.simplify(a2*a1)}*x + {sp.simplify(a2*b1)}*y = {sp.simplify(a2*c1)}")
        steps.append(f"  Multiply (2) by {a1} -> {sp.simplify(a1*a2)}*x + {sp.simplify(a1*b2)}*y = {sp.simplify(a1*c2)}")
        steps.append("  Subtract the second from the first to eliminate x:")
        num_y = sp.simplify(a2*c1 - a1*c2)
        den_y = sp.simplify(a2*b1 - a1*b2)
        steps.append(f"    numerator = {sp.pretty(num_y)}")
        steps.append(f"    denominator = {sp.pretty(den_y)}")
        y_val = sp.simplify(num_y / den_y)
        steps.append(f"  So y = {y_val}")
        steps.append("")
        steps.append("Back-substitute y into equation (1) to find x:")
        x_val = sp.simplify((c1 - b1*y_val) / a1)
        steps.append(f"  x = {x_val}")
        steps.append("")
        steps.append(f"Solution: x = {sp.simplify(x_val)}, y = {sp.simplify(y_val)}")
        return "\n".join(steps)
    except Exception:
        return "Could not generate deterministic steps for this system (parsing error)."

# ---------- LINEAR (single eq) STEPS ----------
def solve_linear_steps(problem_text: str) -> str:
    x = sp.symbols('x')
    s = problem_text.strip()
    s = s.replace("solve for x:", "").replace("Solve for x:", "").replace("solve:", "").replace("Solve:", "")
    s = clean_expr(s)
    if "=" not in s:
        return "Could not parse equation. Provide an equation containing '='."
    try:
        ls, rs = s.split("=", 1)
        L = sp.sympify(ls.strip()); R = sp.sympify(rs.strip())
        eq = sp.Eq(L, R)
        # Attempt to collect x terms
        expr = sp.expand(L - R)
        coeff = sp.simplify(sp.diff(expr, x))  # derivative wrt x -> coefficient of x
        const = sp.simplify(expr.subs(x, 0))
        # Better: solve symbolically to get solution
        sol = sp.solve(eq, x)
        if not sol:
            return "No solution found."
        sol_val = sp.nsimplify(sol[0])
        steps = []
        steps.append(f"Equation: {sp.pretty(eq)}")
        steps.append("Step 1: Move constants to the right-hand side, collect x terms.")
        steps.append(f"  Simplified left - right -> {sp.pretty(expr)}")
        steps.append(f"Step 2: Solve for x: x = {sol_val}")
        return "\n".join(steps)
    except Exception:
        return "Could not parse or solve linear equation."

# ---------- QUADRATIC STEPS ----------
def solve_quadratic_steps(problem_text: str) -> str:
    x = sp.symbols('x')
    s = problem_text.strip()
    s = s.replace("solve:", "").replace("Solve:", "")
    s = clean_expr(s)
    if "=" not in s:
        return "Could not parse equation. Provide an equation containing '='."
    try:
        ls, rs = s.split("=", 1)
        eq = sp.Eq(sp.sympify(ls.strip()), sp.sympify(rs.strip()))
        poly = sp.simplify(sp.expand(eq.lhs - eq.rhs))
        a = sp.simplify(sp.Poly(poly, x).coeff_monomial(x**2))
        b = sp.simplify(sp.Poly(poly, x).coeff_monomial(x))
        c = sp.simplify(sp.Poly(poly, x).coeff_monomial(1))
        steps = []
        steps.append(f"Quadratic: {sp.pretty(eq)} -> standard form: {sp.pretty(poly)} = 0")
        # try factoring
        fac = sp.factor(poly)
        if fac != poly:
            steps.append(f"Factored form: {sp.pretty(fac)}")
            roots = sp.solve(sp.Eq(poly,0), x)
            steps.append("So roots are: " + ", ".join([str(sp.nsimplify(r)) for r in roots]))
            return "\n".join(steps)
        # else quadratic formula
        steps.append("Could not factorize cleanly. Use quadratic formula:")
        steps.append(f"  a = {a}, b = {b}, c = {c}")
        steps.append("  x = (-b ± sqrt(b^2 - 4ac)) / (2a)")
        disc = sp.simplify(b**2 - 4*a*c)
        steps.append(f"  Discriminant = b^2 - 4ac = {sp.pretty(disc)}")
        sqrt_disc = sp.sqrt(disc)
        x1 = sp.nsimplify((-b + sqrt_disc) / (2*a))
        x2 = sp.nsimplify((-b - sqrt_disc) / (2*a))
        steps.append(f"  Solutions: x = {x1} , x = {x2}")
        return "\n".join(steps)
    except Exception:
        return "Could not produce quadratic steps (parsing error)."

# ---------- DERIVATIVE STEPS (human friendly) ----------
def solve_derivative_steps(problem_text: str) -> str:
    x = sp.symbols('x')
    raw = problem_text
    s = re.sub(r'(?i)differentiat(e|ion|e):?', '', raw)
    s = s.replace("f(x) =", "").replace("y =", "")
    s = clean_expr(s)
    try:
        expr = sp.sympify(s)
        der = sp.simplify(sp.diff(expr, x))
        steps = []
        steps.append(f"Differentiate f(x) = {sp.pretty(expr)} with respect to x.")
        # heuristics to describe rule
        text = sp.srepr(expr)
        # chain rule detection: presence of function of something (sin(...), cos(...), etc)
        if any(fn in str(expr) for fn in ("sin(", "cos(", "tan(", "exp(", "log(")):
            steps.append("We detect composed functions (e.g. trig/exp/log). We'll apply the chain rule where needed.")
        # example rule message
        steps.append("Rules applied:")
        steps.append("  - Power rule: d/dx[x^n] = n*x^(n-1)")
        steps.append("  - Product rule: d(uv)/dx = u'v + uv' (if product detected)")
        steps.append("  - Chain rule: d[f(g(x))]/dx = f'(g(x)) * g'(x) (if composition detected)")
        steps.append("")
        steps.append(f"Symbolic derivative: {sp.pretty(der)}")
        return "\n".join(steps)
    except Exception:
        return "Could not compute derivative steps (parsing error)."

# ---------- INTEGRAL STEPS (include trig chain-rule explanation) ----------
def strip_natural_language(text: str) -> str:
    """
    Removes English words and keeps only mathematical content
    """
    # lower for consistency
    text = text.lower()

    # remove common math request words
    text = re.sub(
        r'\b(find|evaluate|calculate|compute|solve|determine|value of|the|expression|of|is)\b',
        '',
        text
    )

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def solve_integral_steps(problem_text: str):
    x = sp.symbols('x')

    # 0️⃣ Remove natural language safely
    text = strip_natural_language(problem_text)

    # 1️⃣ Remove integrate keyword
    text = re.sub(r'(?i)integrat(e|ion|e):?', '', text)

    # 2️⃣ Extract limits (supports pi, expressions, fractions)
    limits = None
    limit_match = re.search(r'on\s*\[\s*([^\],]+)\s*,\s*([^\]]+)\s*\]', text)
    if limit_match:
        a = sp.sympify(clean_expr(limit_match.group(1)))
        b = sp.sympify(clean_expr(limit_match.group(2)))
        limits = (a, b)
        text = re.sub(r'on\s*\[.*?\]', '', text)

    # 3️⃣ Remove function definitions like f(x)= g(x)= h(x)=
    text = re.sub(r'[a-zA-Z]\s*\(\s*x\s*\)\s*=', '', text)

    # 4️⃣ Normalize powers like sin**2(x) → sin(x)**2
    text = re.sub(r'(sin|cos|tan)\*\*2\s*\(([^)]+)\)', r'\1(\2)**2', text)

    # 5️⃣ Final cleanup
    text = text.replace("dx", "").strip()
    text = clean_expr(text)

    try:
        expr = sp.sympify(
            text,
            locals={
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "sqrt": sp.sqrt,
                "pi": sp.pi
            }
        )

        # 🔹 Definite Integral
        if limits:
            a, b = limits
            result = sp.integrate(expr, (x, a, b))
            explanation = (
                "Integrate the function:\n"
                f"∫ {sp.pretty(expr)} dx\n\n"
                f"Limits detected: from {a} to {b}\n\n"
                "Exact result:\n"
                f"{sp.pretty(result)}"
            )
            return explanation, str(result)

        # 🔹 Indefinite Integral
        else:
            result = sp.integrate(expr, x)
            explanation = (
                "Integrate the function:\n"
                f"∫ {sp.pretty(expr)} dx\n\n"
                "Exact result:\n"
                f"{sp.pretty(result)} + C"
            )
            return explanation, str(result) + " + C"

    except Exception as e:
        return f"Could not compute integral steps. Error: {str(e)}", None





    
# ---------- TRIGONOMETRIC IDENTITY VERIFICATION ----------
def solve_trig_identity(problem_text: str):
    A = sp.symbols('A')

    s = problem_text.lower()
    s = s.replace("evaluate:", "").replace("verify:", "").strip()
    s = clean_expr(s)

    if "=" not in s:
        return None, None

    try:
        lhs, rhs = s.split("=", 1)
        lhs_expr = sp.sympify(lhs)
        rhs_expr = sp.sympify(rhs)

        diff = sp.simplify(lhs_expr - rhs_expr)

        steps = []
        steps.append("This is a trigonometric identity verification.")
        steps.append("")
        steps.append("Given:")
        steps.append(f"LHS = {sp.pretty(lhs_expr)}")
        steps.append(f"RHS = {sp.pretty(rhs_expr)}")
        steps.append("")
        steps.append("Subtract RHS from LHS:")
        steps.append(f"LHS − RHS = {sp.pretty(diff)}")
        steps.append("")

        if diff == 0:
            steps.append("Since LHS − RHS = 0, the identity is VERIFIED.")
            return "\n".join(steps), "Identity verified (True)"
        else:
            steps.append("Since LHS − RHS ≠ 0, the identity is NOT true.")
            return "\n".join(steps), "Identity not verified (False)"

    except Exception as e:
        return f"Could not verify identity. Error: {str(e)}", None


# ---------- trig STEPS (include trig chain-rule explanation) ----------

def solve_trig_evaluate(problem_text: str):
    x = sp.symbols('x')

    try:
        # Remove 'evaluate:' keyword
        s = re.sub(r'(?i)evaluate\s*:\s*', '', problem_text)

        # Convert degree symbol to radians
        s = s.replace("°", "*pi/180")

        expr = sp.sympify(s)

        # Exact symbolic result
        exact = sp.simplify(expr)

        # Numerical evaluation
        numeric = exact.evalf(6)

        explanation = []
        explanation.append("This is a trigonometric evaluation.")
        explanation.append("Convert degrees to radians:")
        explanation.append("  θ° = θ × π / 180")
        explanation.append("")
        explanation.append("Evaluate the expression:")
        explanation.append(f"  {sp.pretty(exact)}")
        explanation.append("")
        explanation.append("Numerical approximation:")
        explanation.append(f"  {numeric}")

        return "\n".join(explanation), str(numeric)

    except Exception as e:
        return f"Could not evaluate trigonometric expression. Error: {str(e)}", None



# ---------- SIMPLE RULE-BASED CLASSIFIER ----------
def classify_problem(problem_text: str) -> str:
    text = problem_text.lower()

    # Existing math
    if "integrate" in text or "integral" in text or "∫" in text:
        return "integral"
    if "differentiate" in text or "derivative" in text:
        return "derivative"
    if "system" in text:
        return "system"
    if "^2" in text or "x**2" in text:
        return "quadratic"
    if "x" in text and "=" in text and "system" not in text:
        return "linear"
    if "evaluate" in text and any(fn in text for fn in ["sin", "cos", "tan"]):
        return "trigonometric"

    # 🔹 QUANT CATEGORIES
    if "%" in text or "percent" in text:
        return "percentage"
    if "profit" in text or "loss" in text:
        return "profit_loss"
    if "simple interest" in text:
        return "simple_interest"
    if "compound interest" in text:
        return "compound_interest"
    if "time and work" in text:
        return "time_work"
    if "speed" in text and "distance" in text:
        return "speed_distance"
    if "ratio" in text:
        return "ratio"
    if "average" in text:
        return "average"
    if "age" in text:
        return "ages"
    if "probability" in text:
        return "probability"

    return "unknown"



# ---------- MAIN SOLVER (deterministic explanations + seq2seq fallback) ----------
def solve_math_problem(problem_text: str): 
    category = classify_problem(problem_text)
    explanation = ""
    exact_answer = None

    # ---------- CORE MATH ----------
    if category == "system":
        explanation = solve_system_steps(problem_text)

    elif category == "linear":
        explanation = solve_linear_steps(problem_text)

    elif category == "quadratic":
        explanation = solve_quadratic_steps(problem_text)

    elif category == "derivative":
        explanation = solve_derivative_steps(problem_text)

    elif category == "integral":
        explanation, exact_answer = solve_integral_steps(problem_text)

    elif category == "trigonometric":
        if "=" in problem_text:
            explanation, exact_answer = solve_trig_identity(problem_text)
        else:
            explanation, exact_answer = solve_trig_evaluate(problem_text)

    # ---------- QUANT MODULE ----------
    elif category == "percentage":
        explanation, exact_answer = solve_percentage(problem_text)

    elif category == "profit_loss":
        explanation, exact_answer = solve_profit_loss(problem_text)

    elif category == "simple_interest":
        explanation, exact_answer = solve_simple_interest(problem_text)

    elif category == "compound_interest":
        explanation, exact_answer = solve_compound_interest(problem_text)

    elif category == "time_work":
        explanation, exact_answer = solve_time_work(problem_text)

    elif category == "speed_distance":
        explanation, exact_answer = solve_speed_distance(problem_text)

    elif category == "ratio":
        explanation, exact_answer = solve_ratio(problem_text)

    elif category == "average":
        explanation, exact_answer = solve_average(problem_text)

    elif category == "ages":
        explanation, exact_answer = solve_ages(problem_text)

    elif category == "probability":
        explanation, exact_answer = solve_probability(problem_text)

    # ---------- FALLBACK ----------
    else:
        explanation = decode_sequence(problem_text, max_decoder_len=400)
        if not explanation or len(explanation.strip()) < 3:
            explanation = "Step-by-step explanation not available."

    # Final safety
    if exact_answer is None:
        exact_answer = get_exact_answer(problem_text, category)

    return {
        "problem": problem_text,
        "category": category,
        "explanation": explanation,
        "answer": exact_answer
    }



def solve_percentage(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))

    if len(nums) >= 2:
        price, percent = nums[0], nums[1]

        discount = (percent / 100) * price
        final_price = price - discount

        explanation = (
            f"Original Price = {price}\n"
            f"Discount = {percent}%\n\n"
            f"Discount Amount = ({percent}/100) × {price} = {discount}\n"
            f"Final Price = {price} − {discount} = {final_price}"
        )

        return explanation, final_price

    return "Could not parse percentage problem.", None



def solve_profit_loss(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))
    if len(nums) >= 2:
        cp, sp = nums[0], nums[1]
        if sp > cp:
            profit = sp - cp
            profit_percent = (profit / cp) * 100
            explanation = (
                f"Cost Price = {cp}\n"
                f"Selling Price = {sp}\n"
                f"Profit = SP − CP = {profit}\n"
                f"Profit % = (Profit / CP) × 100\n"
                f"= {profit_percent}%"
            )
            return explanation, f"{profit_percent}% profit"
        else:
            loss = cp - sp
            loss_percent = (loss / cp) * 100
            explanation = (
                f"Loss = CP − SP = {loss}\n"
                f"Loss % = (Loss / CP) × 100\n"
                f"= {loss_percent}%"
            )
            return explanation, f"{loss_percent}% loss"
    return "Could not parse profit/loss problem.", None

def solve_simple_interest(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))
    if len(nums) >= 3:
        P, R, T = nums[:3]
        si = (P * R * T) / 100
        amount = P + si
        explanation = (
            "Simple Interest Formula:\n"
            "SI = (P × R × T) / 100\n\n"
            f"P = {P}, R = {R}%, T = {T}\n"
            f"SI = ({P}×{R}×{T}) / 100 = {si}\n"
            f"Total Amount = {amount}"
        )
        return explanation, amount
    return "Could not parse simple interest problem.", None

def solve_compound_interest(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))
    if len(nums) >= 3:
        P, R, T = nums[:3]
        A = P * (1 + R/100) ** T
        ci = A - P
        explanation = (
            "Compound Interest Formula:\n"
            "A = P(1 + R/100)^T\n\n"
            f"P = {P}, R = {R}%, T = {T}\n"
            f"A = {A}\n"
            f"CI = A − P = {ci}"
        )
        return explanation, ci
    return "Could not parse compound interest problem.", None

def solve_time_work(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))
    if len(nums) >= 2:
        A, B = nums[:2]
        rate = (1/A) + (1/B)
        days = 1 / rate
        explanation = (
            f"A can do work in {A} days\n"
            f"B can do work in {B} days\n\n"
            "Combined rate = 1/A + 1/B\n"
            f"= {rate}\n"
            f"Total time = {days} days"
        )
        return explanation, days
    return "Could not parse time & work problem.", None

def solve_speed_distance(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))
    if len(nums) >= 2:
        distance, time = nums[:2]
        speed = distance / time
        explanation = (
            "Speed Formula:\n"
            "Speed = Distance / Time\n\n"
            f"Speed = {distance} / {time} = {speed}"
        )
        return explanation, speed
    return "Could not parse speed-distance problem.", None


def solve_ratio(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))
    if len(nums) >= 2:
        a, b = nums[:2]
        explanation = (
            "Ratio Formula:\n"
            f"Ratio = {a} : {b}"
        )
        return explanation, f"{a}:{b}"
    return "Could not parse ratio problem.", None


def solve_average(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))
    if nums:
        avg = sum(nums) / len(nums)
        explanation = (
            f"Average = Sum / Count\n"
            f"= {sum(nums)} / {len(nums)}\n"
            f"= {avg}"
        )
        return explanation, avg
    return "Could not parse average problem.", None


def solve_ages(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))
    if len(nums) >= 2:
        present, years = nums[:2]
        future = present + years
        explanation = (
            f"Present age = {present}\n"
            f"After {years} years = {future}"
        )
        return explanation, future
    return "Could not parse age problem.", None


def solve_probability(problem):
    nums = list(map(float, re.findall(r'\d+\.?\d*', problem)))
    if len(nums) >= 2:
        fav, total = nums[:2]
        prob = fav / total
        explanation = (
            "Probability Formula:\n"
            "P(E) = favorable / total\n\n"
            f"P = {fav}/{total} = {prob}"
        )
        return explanation, prob
    return "Could not parse probability problem.", None


# ---------- FLASK ROUTES ----------
from flask import Flask, render_template, request, jsonify, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "secure_key_123"


# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        mode = request.form.get("mode")   # math or quant

        if not name or not email or not mode:
            return render_template("login.html", error="All fields required")

        session["user"] = {
            "name": name,
            "email": email,
            "mode": mode
        }

        if mode == "math":
            return redirect(url_for("math_page"))
        else:
            return redirect(url_for("quant_page"))

    return render_template("login.html")


# ---------------- MATH PAGE ----------------
@app.route("/math")
def math_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session["user"])


@app.route("/solve", methods=["POST"])
def solve_math():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(force=True)
    problem = data.get("problem", "").strip()

    if not problem:
        return jsonify({"error": "Empty problem"}), 400

    return jsonify(solve_math_problem(problem))


# ---------------- QUANT PAGE ----------------
@app.route("/quant")
def quant_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("quant.html", user=session["user"])


@app.route("/solve-quant", methods=["POST"])
def solve_quant():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json(force=True)
    problem = data.get("problem", "").lower().strip()

    if not problem:
        return jsonify({"error": "Empty problem"}), 400

    explanation = None
    answer = None
    topic = "Unknown"

    if any(w in problem for w in ["percent", "%"]):
        explanation, answer = solve_percentage(problem)
        topic = "Percentages"

    elif any(w in problem for w in ["profit", "loss", "cp", "sp"]):
        explanation, answer = solve_profit_loss(problem)
        topic = "Profit & Loss"

    elif "simple interest" in problem:
        explanation, answer = solve_simple_interest(problem)
        topic = "Simple Interest"

    elif "compound interest" in problem:
        explanation, answer = solve_compound_interest(problem)
        topic = "Compound Interest"

    elif "time" in problem and "work" in problem:
        explanation, answer = solve_time_work(problem)
        topic = "Time & Work"

    elif any(w in problem for w in ["speed", "distance"]):
        explanation, answer = solve_speed_distance(problem)
        topic = "Time, Speed & Distance"

    elif "ratio" in problem:
        explanation, answer = solve_ratio(problem)
        topic = "Ratio & Proportion"

    elif "average" in problem:
        explanation, answer = solve_average(problem)
        topic = "Averages"

    elif "age" in problem:
        explanation, answer = solve_ages(problem)
        topic = "Ages"

    elif "probability" in problem:
        explanation, answer = solve_probability(problem)
        topic = "Probability"

    else:
        explanation = "Unsupported quantitative topic."

    return jsonify({
        "topic": topic,
        "explanation": explanation,
        "answer": answer
    })


# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)

