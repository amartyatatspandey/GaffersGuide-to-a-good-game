# import pymysql
# from flask import Flask, render_template, request, redirect, url_for, session, flash
# from pathlib import Path
# import traceback
# import os
# import shutil
# import datetime
# import logging
# from werkzeug.security import generate_password_hash, check_password_hash
# def get_db():
#     return pymysql.connect(
#         host="localhost",
#         user="root",
#         password="anshul2005@", 
#         database="social_media"
#     )
# # import model wrappers
# from models.model1 import model_main as model1_main
# from models.model2 import model_main as model2_main
# from models.model3 import model_main as model3_main

# # -------------------------------------------------------
# # BASIC CONFIG
# # -------------------------------------------------------

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__, static_folder="static", template_folder="templates")
# app.secret_key = "dev-secret-replace-me"

# ALLOWED_COPY_EXT = (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".csv", ".json")
# MAX_FILES_TO_COPY = 200

# PROJECT_ROOT = Path(__file__).resolve().parent
# STATIC_PLOTS = PROJECT_ROOT / "static" / "plots"
# STATIC_PLOTS.mkdir(parents=True, exist_ok=True)

# # -------------------------------------------------------
# # SIMPLE USER STORE (demo)
# # -------------------------------------------------------
# USERS = {}


# def ensure_result_dict(result):
#     if isinstance(result, dict):
#         return result
#     return {"result": str(result)}


# # -------------------------------------------------------
# # AUTH ROUTES
# # -------------------------------------------------------

# @app.route('/')
# def index():
#     if session.get('user'):
#         return redirect(url_for('home'))
#     return render_template('index.html')


# # -------------------------------------------------------
# # SIGNUP ROUTE (restored)
# # -------------------------------------------------------
# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         name = request.form.get('name', '').strip()
#         email = request.form.get('email', '').lower().strip()
#         pwd = request.form.get('password', '')

#         if not name or not email or not pwd:
#             flash("Please fill all fields", "error")
#             return redirect(url_for('signup'))

#         conn = get_db()
#         cur = conn.cursor()
#         try:
#             cur.execute(
#                 "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
#                 (name, email, generate_password_hash(pwd))
#             )
#             conn.commit()
#             user_id = cur.lastrowid
#         except IntegrityError:
#             # duplicate email
#             conn.rollback()
#             flash("Email already registered", "error")
#             return redirect(url_for('signup'))
#         except Exception as e:
#             conn.rollback()
#             logger.exception("DB error during signup: %s", e)
#             flash("Database error. See server log.", "error")
#             return redirect(url_for('signup'))
#         finally:
#             cur.close()
#             conn.close()

#         # save id + name in session
#         session['user_id'] = int(user_id)
#         session['user'] = name
#         flash("Account created", "success")
#         return redirect(url_for('home'))

#     return render_template('signup.html')


# import pymysql
# # ensure you have this at top (you already import pymysql)
# # import pymysql.err and cursors for clarity
# from pymysql.err import IntegrityError
# from pymysql.cursors import DictCursor

# def get_db():
#     # return a connection that yields dict rows by default
#     return pymysql.connect(
#         host="localhost",
#         user="root",
#         password="anshul2005@",
#         database="social_media",
#         cursorclass=DictCursor,
#         autocommit=False
#     )

# # -------- SIGNUP (single, clean implementation) ----------
# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         name = request.form.get('name', '').strip()
#         email = request.form.get('email', '').lower().strip()
#         pwd = request.form.get('password', '')

#         if not name or not email or not pwd:
#             flash("Please fill all fields", "error")
#             return redirect(url_for('signup'))

#         conn = get_db()
#         cur = conn.cursor()
#         try:
#             cur.execute(
#                 "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
#                 (name, email, generate_password_hash(pwd))
#             )
#             conn.commit()
#             user_id = cur.lastrowid
#         except IntegrityError:
#             # duplicate email
#             conn.rollback()
#             flash("Email already registered", "error")
#             return redirect(url_for('signup'))
#         except Exception as e:
#             conn.rollback()
#             logger.exception("DB error during signup: %s", e)
#             flash("Database error. See server log.", "error")
#             return redirect(url_for('signup'))
#         finally:
#             cur.close()
#             conn.close()

#         # save id + name in session
#         session['user_id'] = int(user_id)
#         session['user'] = name
#         flash("Account created", "success")
#         return redirect(url_for('home'))

#     return render_template('signup.html')


# # -------- LOGIN (clean MySQL-backed version) ----------
# @app.route('/login', methods=['GET','POST'])
# def login():
#     if request.method == 'POST':
#         email = request.form.get('email', '').lower().strip()
#         pwd = request.form.get('password', '')

#         conn = get_db()
#         cur = conn.cursor()
#         try:
#             cur.execute("SELECT id, name, password_hash FROM users WHERE email = %s", (email,))
#             user = cur.fetchone()
#         except Exception:
#             logger.exception("DB error on login")
#             flash('Database error', 'error')
#             return redirect(url_for('login'))
#         finally:
#             cur.close()
#             conn.close()

#         if not user or not check_password_hash(user['password_hash'], pwd):
#             flash('Invalid email or password', 'error')
#             return redirect(url_for('login'))

#         session['user_id'] = int(user['id'])
#         session['user'] = user['name']
#         flash('Signed in', 'success')
#         return redirect(url_for('home'))

#     return render_template('login.html')


       


# @app.route('/logout')
# def logout():
#     session.clear()
#     flash("Logged out", "success")
#     return redirect(url_for('index'))


# @app.route('/home')
# def home():
#     if not session.get('user'):
#         flash("Please sign in", "error")
#         return redirect(url_for('login'))
#     return render_template('home.html')


# # -------------------------------------------------------
# # RESULTS PAGE (protected)
# # -------------------------------------------------------

# @app.route('/results')
# def results():
#     if not session.get('user'):
#         flash("Please sign in to view results", "error")
#         return redirect(url_for('login'))

#     return render_template(
#         "results.html",
#         model_id="",
#         match_path="",
#         result={}
#     )


# # -------------------------------------------------------
# # MODEL RUNNER
# # -------------------------------------------------------

# @app.route("/run_model/<model_id>", methods=["GET", "POST"])
# def run_model(model_id):
#     if not session.get("user"):
#         flash("Login required", "error")
#         return redirect(url_for("login"))

#     match_path = request.values.get("match_path", "").strip()
#     if not match_path:
#         return "Error: Provide match_path", 400

#     try:
#         if model_id == "1":
#             raw_result = model1_main(match_path)
#         elif model_id == "2":
#             raw_result = model2_main(match_path)
#         elif model_id == "3":
#             raw_result = model3_main(match_path)
#         else:
#             return "Unknown model id", 404
#     except Exception:
#         tb = traceback.format_exc()
#         logger.exception("Model error")
#         return f"<h3>Model error</h3><pre>{tb}</pre>", 500

#     result = ensure_result_dict(raw_result)

#     output_folder = result.get("output_folder")
#     if not output_folder or not os.path.exists(output_folder):
#         result["served_plots"] = []
#         result["served_output_dir"] = ""
#         result["error"] = f"Output not found: {output_folder}"
#         return render_template("results.html", model_id=model_id, match_path=match_path, result=result)

#     run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
#     dest_dir = STATIC_PLOTS / run_id
#     dest_dir.mkdir(parents=True, exist_ok=True)

#     copied = []
#     for fname in sorted(os.listdir(output_folder)):
#         if fname.lower().endswith(ALLOWED_COPY_EXT):
#             src = os.path.join(output_folder, fname)
#             if os.path.isfile(src):
#                 shutil.copy2(src, dest_dir)
#                 copied.append(fname)
#                 if len(copied) >= MAX_FILES_TO_COPY:
#                     break

#     public_urls = [url_for("static", filename=f"plots/{run_id}/{f}") for f in copied]

#     result["served_plots"] = public_urls
#     result["served_output_dir"] = f"static/plots/{run_id}"

#     return render_template("results.html", model_id=model_id, match_path=match_path, result=result)


# # -------------------------------------------------------
# # RUN
# # -------------------------------------------------------

# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5001, debug=True)

import pymysql
from pymysql.err import IntegrityError
from pymysql.cursors import DictCursor

from flask import Flask, render_template, request, redirect, url_for, session, flash
from pathlib import Path
import traceback
import os
import shutil
import datetime
import logging
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------------

def get_db():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="anshul2005@",
        database="social_media",
        cursorclass=DictCursor,
        autocommit=False
    )

# -------------------------------------------------------
# MODEL IMPORTS
# -------------------------------------------------------

from models.model1 import model_main as model1_main
from models.model2 import model_main as model2_main
from models.model3 import model_main as model3_main

# -------------------------------------------------------
# BASIC CONFIG
# -------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "dev-secret-replace-me"

ALLOWED_COPY_EXT = (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".csv", ".json")
MAX_FILES_TO_COPY = 200

PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_PLOTS = PROJECT_ROOT / "static" / "plots"
STATIC_PLOTS.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def ensure_result_dict(result):
    if isinstance(result, dict):
        return result
    return {"result": str(result)}


# -------------------------------------------------------
# AUTH ROUTES
# -------------------------------------------------------

@app.route('/')
def index():
    if session.get('user'):
        return redirect(url_for('home'))
    return render_template('index.html')


# -------------------- SIGNUP ---------------------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').lower().strip()
        pwd = request.form.get('password', '')

        if not name or not email or not pwd:
            flash("Please fill all fields", "error")
            return redirect(url_for('signup'))

        conn = get_db()
        cur = conn.cursor()

        try:
            cur.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
                (name, email, generate_password_hash(pwd))
            )
            conn.commit()
            user_id = cur.lastrowid

        except IntegrityError:
            conn.rollback()
            flash("Email already registered", "error")
            return redirect(url_for('signup'))

        except Exception as e:
            conn.rollback()
            logger.exception("Signup DB error: %s", e)
            flash("Database error", "error")
            return redirect(url_for('signup'))

        finally:
            cur.close()
            conn.close()

        session['user_id'] = int(user_id)
        session['user'] = name
        flash("Account created", "success")
        return redirect(url_for('home'))

    return render_template('signup.html')


# -------------------- LOGIN ----------------------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        pwd = request.form.get('password', '')

        conn = get_db()
        cur = conn.cursor()

        try:
            cur.execute("SELECT id, name, password_hash FROM users WHERE email = %s", (email,))
            user = cur.fetchone()

        except Exception as e:
            logger.exception("Login DB error: %s", e)
            flash("Database error", "error")
            return redirect(url_for('login'))

        finally:
            cur.close()
            conn.close()

        if not user or not check_password_hash(user['password_hash'], pwd):
            flash("Invalid email or password", "error")
            return redirect(url_for('login'))

        session['user_id'] = int(user['id'])
        session['user'] = user['name']
        flash("Signed in", "success")
        return redirect(url_for('home'))

    return render_template('login.html')


# -------------------- LOGOUT ----------------------------

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out", "success")
    return redirect(url_for('index'))


# -------------------- HOME ------------------------------

@app.route('/home')
def home():
    if not session.get('user'):
        flash("Please sign in", "error")
        return redirect(url_for('login'))
    return render_template('home.html')


# -------------------------------------------------------
# RESULTS PAGE
# -------------------------------------------------------

@app.route('/results')
def results():
    if not session.get('user'):
        flash("Please sign in to view results", "error")
        return redirect(url_for('login'))

    return render_template("results.html", model_id="", match_path="", result={})


# -------------------------------------------------------
# MODEL RUNNER
# -------------------------------------------------------

@app.route("/run_model/<model_id>", methods=["GET", "POST"])
def run_model(model_id):
    if not session.get("user"):
        flash("Login required", "error")
        return redirect(url_for("login"))

    match_path = request.values.get("match_path", "").strip()
    if not match_path:
        return "Error: Provide match_path", 400

    try:
        if model_id == "1":
            raw_result = model1_main(match_path)
        elif model_id == "2":
            raw_result = model2_main(match_path)
        elif model_id == "3":
            raw_result = model3_main(match_path)
        else:
            return "Unknown model id", 404

    except Exception:
        tb = traceback.format_exc()
        logger.exception("Model error")
        return f"<h3>Model error</h3><pre>{tb}</pre>", 500

    result = ensure_result_dict(raw_result)

    output_folder = result.get("output_folder")
    if not output_folder or not os.path.exists(output_folder):
        result["served_plots"] = []
        result["served_output_dir"] = ""
        result["error"] = f"Output not found: {output_folder}"
        return render_template("results.html", model_id=model_id, match_path=match_path, result=result)

    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest_dir = STATIC_PLOTS / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for fname in sorted(os.listdir(output_folder)):
        if fname.lower().endswith(ALLOWED_COPY_EXT):
            src = os.path.join(output_folder, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dest_dir)
                copied.append(fname)
                if len(copied) >= MAX_FILES_TO_COPY:
                    break

    public_urls = [
        url_for("static", filename=f"plots/{run_id}/{f}")
        for f in copied
    ]

    result["served_plots"] = public_urls
    result["served_output_dir"] = f"static/plots/{run_id}"

    return render_template("results.html", model_id=model_id, match_path=match_path, result=result)


# -------------------------------------------------------
# RUN APP
# -------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
