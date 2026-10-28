"""
app.py  -  Face Recognition Attendance System

Two completely separate panels:

  ADMIN PANEL     /admin/...   (username + password)
      approvals, employees, departments, attendance reports, announcements, settings, dataset zip,
      admins (the MAIN admin "muzammil" adds permanent/temporary admins; ONLY he can remove
      employees or admins)

  EMPLOYEE PANEL  /login ... /dashboard
      1. Register (name, department, username, password)
      2. Face enrolment (samples are saved into dataset/ and packed into dataset.zip)
      3. "Waiting for admin approval" screen
      4. Admin approves
      5. Login = password + face recognition   ->  Dashboard
      6. Check-in / Check-out (each needs face recognition) -> working hours tracked
      7. An employee can only see their OWN department's information
"""
import csv
import io
import os
import secrets
import threading
import time
import re
from collections import defaultdict
from datetime import timedelta
from functools import wraps

import cv2
from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from werkzeug.utils import secure_filename

import database as db
import face_utils as fu

# ------------------------------------------------------------------ app setup
app = Flask(__name__)


def _load_secret_key():
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = os.path.join(db.DATA_DIR, "secret.key")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(secrets.token_hex(32))
    with open(path) as fh:
        return fh.read().strip()


app.secret_key = _load_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    MAX_CONTENT_LENGTH=4 * 1024 * 1024,
)

db.init_db()
if not os.path.exists(fu.ZIP_PATH):
    fu.rebuild_dataset_zip(db.list_employees())

# access length choices when the main admin adds an admin: key -> (label, days; 0 = permanent)
ADMIN_DURATIONS = {
    "permanent": ("Permanent", 0),
    "1": ("Temporary · 1 day", 1),
    "3": ("Temporary · 3 days", 3),
    "7": ("Temporary · 7 days", 7),
    "30": ("Temporary · 30 days", 30),
}

MAX_FACE_FAILS = 25            # mismatching frames before the attempt is locked
PRE_AUTH_SECONDS = 10 * 60     # time allowed between password step and face step
_enroll_lock = threading.Lock()


# ------------------------------------------------------------------ security helpers
def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def csrf_protect():
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        sent = request.headers.get("X-CSRFToken") or request.form.get("csrf_token", "")
        if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
            abort(400, "Security token missing or expired. Please go back, refresh the page and try again.")


@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(self)")
    if request.endpoint not in ("static",):
        resp.headers["Cache-Control"] = "no-store"
    return resp


_fails = defaultdict(list)


def throttled(key, limit=5, window=600):
    t = time.time()
    _fails[key] = [x for x in _fails[key] if t - x < window]
    return len(_fails[key]) >= limit


def register_fail(key):
    _fails[key].append(time.time())


def clear_fails(key):
    _fails.pop(key, None)


# ------------------------------------------------------------------ jinja helpers
@app.template_filter("hm")
def hm_filter(seconds):
    return db.fmt_hm(seconds)


@app.template_filter("dt")
def dt_filter(value):
    if not value:
        return "—"
    try:
        return db._parse(value).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return value


@app.context_processor
def inject_globals():
    ctx = {"me": g.get("emp"), "admin": g.get("admin"), "SAMPLES_NEEDED": fu.SAMPLES_NEEDED}
    if g.get("admin"):
        ctx["pending_count"] = db.count_status("pending")
        ctx["is_main"] = bool(g.admin.get("is_main"))
    return ctx


# ------------------------------------------------------------------ auth decorators
def admin_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if session.get("role") != "admin":
            return redirect(url_for("admin_login"))
        g.admin = db.get_admin(session.get("admin_id"))
        if not g.admin:                      # removed by the main admin
            session.clear()
            return redirect(url_for("admin_login"))
        if db.admin_expired(g.admin):        # temporary admin whose time is over
            session.clear()
            flash("Your temporary admin access has expired.", "error")
            return redirect(url_for("admin_login"))
        return f(*a, **k)
    return wrapper


def main_admin_required(f):
    """Only the main admin (muzammil). Used for adding/removing admins and removing employees."""
    @wraps(f)
    @admin_required
    def wrapper(*a, **k):
        if not g.admin.get("is_main"):
            abort(403, "Only the main admin can do this.")
        return f(*a, **k)
    return wrapper


def employee_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if session.get("role") != "employee":
            return redirect(url_for("login"))
        emp = db.get_employee(session.get("employee_id"))
        if not emp or emp["status"] != "approved":
            session.clear()
            flash("Your access is no longer active. Please contact the admin.", "error")
            return redirect(url_for("login"))
        g.emp = emp
        return f(*a, **k)
    return wrapper


def pending_employee():
    """Employee who registered/logged in but is not yet fully approved."""
    emp_id = session.get("pending_id")
    return db.get_employee(emp_id) if emp_id else None


def pre_auth_employee():
    """Employee who passed the password step and now needs the face step."""
    emp_id = session.get("pre_auth_id")
    if not emp_id or time.time() - session.get("pre_auth_at", 0) > PRE_AUTH_SECONDS:
        return None
    emp = db.get_employee(emp_id)
    return emp if emp and emp["status"] == "approved" else None


def dataset_manifest():
    return db.list_employees()


# =================================================================== PUBLIC / EMPLOYEE AUTH
@app.route("/")
def index():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    if session.get("role") == "employee":
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    departments = db.list_departments()
    form = {}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip() for k in
                ("full_name", "username", "email", "designation", "department_id")}
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        err = None
        if not form["full_name"] or not form["username"] or not form["department_id"]:
            err = "Name, username and department are required."
        elif len(form["username"]) < 3 or not form["username"].replace("_", "").replace(".", "").isalnum():
            err = "Username must be 3+ characters (letters, numbers, . or _)."
        elif len(password) < 6:
            err = "Password must be at least 6 characters."
        elif password != confirm:
            err = "Passwords do not match."
        elif not db.get_department(int(form["department_id"] or 0)):
            err = "Please select a valid department."
        elif db.username_taken(form["username"]):
            err = "This username is already taken."
        if err:
            flash(err, "error")
        else:
            emp_id = db.create_employee(form["full_name"], form["username"], password,
                                        int(form["department_id"]), form["email"], form["designation"])
            session.clear()
            session["pending_id"] = emp_id
            flash("Account created. Now register your face.", "success")
            return redirect(url_for("enroll"))
    return render_template("register.html", departments=departments, form=form)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("role") == "employee":
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        key = f"emp:{request.remote_addr}:{username.lower()}"
        if throttled(key):
            flash("Too many failed attempts. Please wait 10 minutes and try again.", "error")
            return render_template("login.html"), 429
        emp = db.get_employee_by_username(username)
        if not emp or not db.verify_password(emp, password):
            register_fail(key)
            flash("Invalid username or password.", "error")
            return render_template("login.html")
        clear_fails(key)
        session.clear()
        status = emp["status"]
        if status == "face_pending":
            session["pending_id"] = emp["id"]
            flash("Please finish registering your face.", "success")
            return redirect(url_for("enroll"))
        if status == "pending":
            session["pending_id"] = emp["id"]
            return redirect(url_for("waiting"))
        if status == "rejected":
            flash("Your account was rejected or its access was revoked by the admin.", "error")
            return render_template("login.html")
        session["pre_auth_id"] = emp["id"]
        session["pre_auth_at"] = time.time()
        return redirect(url_for("login_face"))
    return render_template("login.html")


@app.route("/login/face")
def login_face():
    emp = pre_auth_employee()
    if not emp:
        flash("Please enter your username and password first.", "error")
        return redirect(url_for("login"))
    return render_template("face_scan.html", layout="public_base.html", purpose="login",
                           title="Face Verification", subject=emp, early=False, remaining=0)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ------------------------------------------------------------------ enrolment + waiting
@app.route("/enroll")
def enroll():
    emp = pending_employee()
    if not emp:
        return redirect(url_for("login"))
    if emp["status"] == "pending":
        return redirect(url_for("waiting"))
    if emp["status"] != "face_pending":
        return redirect(url_for("login"))
    fu.reset_samples(emp["id"])          # always start a clean capture
    db.set_samples(emp["id"], 0)
    return render_template("enroll.html", emp=emp, needed=fu.SAMPLES_NEEDED)


@app.route("/api/enroll/frame", methods=["POST"])
def api_enroll_frame():
    emp = pending_employee()
    if not emp or emp["status"] != "face_pending":
        return jsonify(ok=False, error="Session expired. Please login again."), 403
    frame = fu.decode_image((request.get_json(silent=True) or {}).get("image", ""))
    if frame is None:
        return jsonify(ok=False, error="Invalid image."), 400
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = fu.detect_faces(gray)
    count = fu.count_samples(emp["id"])
    if len(faces) == 0:
        return jsonify(ok=True, state="no_face", count=count)
    x, y, w, h = [int(v) for v in fu.largest_face(faces)]
    if any(f[2] * f[3] > 0.5 * w * h and tuple(int(v) for v in f) != (x, y, w, h) for f in faces):
        return jsonify(ok=True, state="multiple_faces", count=count)

    with _enroll_lock:
        count = fu.count_samples(emp["id"])
        if count >= fu.SAMPLES_NEEDED:
            return jsonify(ok=True, state="done", count=count, done=True, redirect=url_for("waiting"))
        count = fu.save_face_sample(emp["id"], gray[y:y + h, x:x + w])
        if count < fu.SAMPLES_NEEDED:
            return jsonify(ok=True, state="captured", count=count, box=[x, y, w, h],
                           frame=[frame.shape[1], frame.shape[0]])
        # ---- enrolment complete ----
        similar = fu.find_similar_existing(emp["id"])
        note = ""
        if similar:
            other = db.get_employee(similar[0])
            if other:
                note = (f"Face looks {int(similar[1] * 100)}% similar to existing employee "
                        f"'{other['full_name']}' (#{other['id']}). Please verify identity before approving.")
        ok, msg = fu.train_model()
        if not ok:
            return jsonify(ok=False, error=msg), 500
        db.set_samples(emp["id"], count)
        db.set_status(emp["id"], "pending", flag_note=note)
        fu.rebuild_dataset_zip(dataset_manifest())
    return jsonify(ok=True, state="done", count=count, done=True, redirect=url_for("waiting"))


@app.route("/waiting")
def waiting():
    emp = pending_employee()
    if not emp:
        return redirect(url_for("login"))
    if emp["status"] == "face_pending":
        return redirect(url_for("enroll"))
    return render_template("waiting.html", emp=emp)


@app.route("/api/status")
def api_status():
    emp = pending_employee()
    if not emp:
        return jsonify(status="none"), 401
    return jsonify(status=emp["status"])


@app.route("/waiting/continue")
def waiting_continue():
    """After approval the employee must log in normally (password + face)."""
    session.clear()
    return redirect(url_for("login"))


# =================================================================== FACE VERIFY (login + check in/out)
@app.route("/api/face/verify", methods=["POST"])
def api_face_verify():
    data = request.get_json(silent=True) or {}
    purpose = data.get("purpose")
    if purpose == "login":
        emp = pre_auth_employee()
    elif purpose in ("checkin", "checkout") and session.get("role") == "employee":
        emp = db.get_employee(session.get("employee_id"))
        if emp and emp["status"] != "approved":
            emp = None
    else:
        return jsonify(ok=False, error="Bad request."), 400
    if not emp:
        return jsonify(ok=False, state="auth", error="Session expired. Please login again.",
                       redirect=url_for("login")), 401

    frame = fu.decode_image(data.get("image", ""))
    if frame is None:
        return jsonify(ok=False, error="Invalid image."), 400
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res = fu.verify_face(gray, emp["id"])
    out = {"ok": True, "state": res["state"], "needed": fu.STREAK_NEEDED,
           "box": res.get("box"), "frame": [frame.shape[1], frame.shape[0]]}

    if res["state"] in ("no_face", "multiple_faces", "no_model"):
        out["streak"] = (session.get("face_state") or {}).get("streak", 0) if res["state"] != "multiple_faces" else 0
        return jsonify(out)

    st = session.get("face_state") or {}
    if st.get("purpose") != purpose or st.get("emp") != emp["id"]:
        st = {"purpose": purpose, "emp": emp["id"], "streak": 0, "fails": 0}
    if res["match"]:
        st["streak"] += 1
    else:
        st["streak"], st["fails"] = 0, st["fails"] + 1

    if st["fails"] >= MAX_FACE_FAILS:
        session.pop("face_state", None)
        if purpose == "login":
            session.pop("pre_auth_id", None)
            out.update(state="locked", error="Face did not match. Please login again.", redirect=url_for("login"))
        else:
            out.update(state="locked", error="Face did not match. Please try again.")
        return jsonify(out)

    out["streak"] = st["streak"]
    if st["streak"] < fu.STREAK_NEEDED:
        session["face_state"] = st
        return jsonify(out)

    # ----- verified: perform the action right here (no client-side flag to forge) -----
    session.pop("face_state", None)
    if purpose == "login":
        session.clear()
        session.permanent = True
        session["role"] = "employee"
        session["employee_id"] = emp["id"]
        out.update(state="verified", message=f"Welcome, {emp['full_name']}!", redirect=url_for("dashboard"))
    elif purpose == "checkin":
        ok, msg = db.check_in(emp["id"])
        if ok:
            choice = session.pop("attendance_choice", {})
            with db.conn_ctx() as c:
                c.execute("UPDATE attendance SET work_mode=?, location_name=?, latitude=?, longitude=? WHERE employee_id=? AND work_date=?", (choice.get("mode", "Office"), choice.get("location", ""), choice.get("latitude"), choice.get("longitude"), emp["id"], db.today_str()))
        flash(msg, "success" if ok else "error")
        out.update(state="verified" if ok else "blocked", message=msg, redirect=url_for("dashboard"))
    else:
        ok, msg = db.check_out(emp["id"], allow_early=bool(data.get("early_ok")))
        flash(msg, "success" if ok else "error")
        out.update(state="verified" if ok else "blocked", message=msg, redirect=url_for("dashboard"))
    return jsonify(out)


# =================================================================== EMPLOYEE PANEL
@app.route("/dashboard")
@employee_required
def dashboard():
    emp = g.emp
    settings = db.get_settings()
    req = db.required_seconds(settings)
    rec = db.get_today_record(emp["id"])
    today = db.enrich(rec, req) if rec else None
    return render_template(
        "employee/dashboard.html", active="dashboard", settings=settings, req=req, today=today,
        summary=db.month_summary(emp["id"]), history=db.employee_history(emp["id"], limit=7),
        announcements=db.list_announcements(emp["department_id"], limit=3),
        server_now=db.now().strftime(db.DT_FMT),
    )


@app.route("/attendance/scan/<action>")
@employee_required
def scan(action):
    if action not in ("checkin", "checkout"):
        abort(404)
    emp = g.emp
    rec = db.get_today_record(emp["id"])
    req = db.required_seconds()
    early, remaining = False, 0
    if action == "checkin" and rec:
        flash("You have already checked in today.", "error")
        return redirect(url_for("dashboard"))
    if action == "checkout":
        if not rec:
            flash("You have not checked in today.", "error")
            return redirect(url_for("dashboard"))
        if rec["check_out"]:
            flash("You have already checked out today.", "error")
            return redirect(url_for("dashboard"))
        e = db.enrich(rec, req)
        if e["worked"] < req:
            early, remaining = True, req - e["worked"]
    if action == "checkin" and not session.get("attendance_choice"):
        return redirect(url_for("attendance_setup"))
    return render_template("face_scan.html", layout="employee/base.html", active="dashboard",
                           purpose=action, subject=emp, early=early, remaining=remaining,
                           title="Check In" if action == "checkin" else "Check Out")


@app.route("/my-attendance")
@employee_required
def my_attendance():
    month = request.args.get("month") or db.now().strftime("%Y-%m")
    return render_template("employee/attendance.html", active="attendance", month=month,
                           rows=db.employee_history(g.emp["id"], month),
                           summary=db.month_summary(g.emp["id"], month))


@app.route("/department")
@employee_required
def my_department():
    return redirect(url_for("department", dept_id=g.emp["department_id"]))


@app.route("/department/<int:dept_id>")
@employee_required
def department(dept_id):
    if dept_id != g.emp["department_id"]:
        abort(403)                       # employees only ever see THEIR department
    return render_template("employee/department.html", active="department",
                           dept=db.get_department(dept_id), members=db.department_members(dept_id),
                           announcements=db.list_announcements(dept_id, limit=20))


@app.route("/profile", methods=["GET", "POST"])
@employee_required
def profile():
    if request.method == "POST":
        action = request.form.get("action", "profile")
        if action == "password":
            cur, new, conf = (request.form.get(k, "") for k in ("current", "new", "confirm"))
            if not db.verify_password(g.emp, cur):
                flash("Current password is wrong.", "error")
            elif len(new) < 8:
                flash("Use at least 8 characters for your new password.", "error")
            elif new != conf:
                flash("New passwords do not match.", "error")
            else:
                db.set_employee_password(g.emp["id"], new)
                flash("Password updated.", "success")
            return redirect(url_for("profile"))
        fields = {
            "email": request.form.get("email", "").strip()[:254],
            "phone": request.form.get("phone", "").strip()[:40],
            "designation": request.form.get("designation", "").strip()[:100],
            "social_linkedin": request.form.get("social_linkedin", "").strip()[:500],
            "social_facebook": request.form.get("social_facebook", "").strip()[:500],
            "social_instagram": request.form.get("social_instagram", "").strip()[:500],
            "social_x": request.form.get("social_x", "").strip()[:500],
            "social_website": request.form.get("social_website", "").strip()[:500],
        }
        # Only accept ordinary web links for social fields.
        for key in ("social_linkedin", "social_facebook", "social_instagram", "social_x", "social_website"):
            val = fields[key]
            if val and not re.match(r"^https?://", val, re.I):
                fields[key] = ""
        upload = request.files.get("profile_picture")
        if upload and upload.filename:
            ext = os.path.splitext(secure_filename(upload.filename))[1].lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                flash("Profile picture must be JPG, PNG, or WEBP.", "error")
                return redirect(url_for("profile"))
            try:
                from PIL import Image
                img = Image.open(upload.stream)
                img.verify()
                upload.stream.seek(0)
            except Exception:
                flash("That image could not be verified. Please upload a valid picture.", "error")
                return redirect(url_for("profile"))
            upload_dir = os.path.join(db.DATA_DIR, "profile_uploads")
            os.makedirs(upload_dir, exist_ok=True)
            filename = f"employee_{g.emp['id']}{ext}"
            # Remove an old image with a different extension.
            for old_ext in (".jpg", ".jpeg", ".png", ".webp"):
                old_path = os.path.join(upload_dir, f"employee_{g.emp['id']}{old_ext}")
                if os.path.exists(old_path) and old_ext != ext:
                    os.remove(old_path)
            upload.save(os.path.join(upload_dir, filename))
            fields["profile_picture"] = filename
        with db.conn_ctx() as c:
            sets = ", ".join(f"{k}=?" for k in fields)
            c.execute(f"UPDATE employees SET {sets} WHERE id=?", (*fields.values(), g.emp["id"]))
        flash("Your profile has been updated.", "success")
        return redirect(url_for("profile"))
    return render_template("employee/profile.html", active="profile",
                           dept=db.get_department(g.emp["department_id"]))


@app.route("/profile-picture/<int:emp_id>")
def profile_picture(emp_id):
    role = session.get("role")
    if role == "employee":
        viewer = db.get_employee(session.get("employee_id"))
        if not viewer or viewer.get("status") != "approved":
            abort(403)
    elif role == "admin":
        viewer = db.get_admin(session.get("admin_id"))
        if not viewer or db.admin_expired(viewer):
            abort(403)
    else:
        abort(403)
    person = db.get_employee(emp_id)
    if not person or person.get("status") != "approved" or not person.get("profile_picture"):
        abort(404)
    filename = person["profile_picture"]
    if os.path.basename(filename) != filename or not filename.startswith(f"employee_{emp_id}."):
        abort(404)
    path = os.path.join(db.DATA_DIR, "profile_uploads", filename)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype={"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(filename.rsplit(".",1)[-1].lower(),"application/octet-stream"))


@app.route("/directory")
@employee_required
def employee_directory():
    q = request.args.get("q", "").strip().lower()
    people = [e for e in db.list_employees(status="approved")
              if not q or q in (e.get("full_name") or "").lower()
              or q in (e.get("department_name") or "").lower()
              or q in (e.get("designation") or "").lower()]
    return render_template("employee/directory.html", active="directory", people=people, q=q)


@app.route("/directory/<int:emp_id>")
@employee_required
def employee_public_profile(emp_id):
    person = db.get_employee(emp_id)
    if not person or person.get("status") != "approved":
        abort(404)
    return render_template("employee/public_profile.html", active="directory", person=person)


@app.route("/tasks", methods=["GET", "POST"])
@employee_required
def employee_tasks():
    if request.method == "POST":
        title = request.form.get("title", "").strip()[:180]
        description = request.form.get("description", "").strip()[:4000]
        if not title:
            flash("Please enter a task or work proposal.", "error")
        else:
            stamp = db.now().strftime(db.DT_FMT)
            with db.conn_ctx() as c:
                c.execute("""INSERT INTO tasks(employee_id,title,description,assigned_by,due_date,status,created_at,updated_at)
                    VALUES (?,?,?,'Employee proposal',?,'Proposed',?,?)""",
                    (g.emp["id"], title, description, request.form.get("due_date", "")[:10], stamp, stamp))
            flash("Your work proposal was sent to admin for review.", "success")
        return redirect(url_for("employee_tasks"))
    with db.conn_ctx() as c:
        tasks = [dict(r) for r in c.execute("SELECT * FROM tasks WHERE employee_id=? ORDER BY created_at DESC", (g.emp["id"],))]
    return render_template("employee/tasks.html", active="tasks", tasks=tasks)


@app.route("/tasks/<int:task_id>/update", methods=["POST"])
@employee_required
def employee_task_update(task_id):
    status = request.form.get("status", "")
    note = request.form.get("employee_note", "").strip()[:2000]
    if status not in ("Assigned", "In Progress", "Completed"):
        abort(400)
    with db.conn_ctx() as c:
        cur = c.execute("UPDATE tasks SET status=?, employee_note=?, updated_at=? WHERE id=? AND employee_id=? AND status IN ('Assigned','In Progress')",
                        (status, note, db.now().strftime(db.DT_FMT), task_id, g.emp["id"]))
        if not cur.rowcount:
            flash("That task cannot be updated.", "error")
        else:
            flash("Task updated.", "success")
    return redirect(url_for("employee_tasks"))


@app.route("/admin/directory")
@admin_required
def admin_directory():
    q = request.args.get("q", "").strip().lower()
    people = [e for e in db.list_employees(status="approved")
              if not q or q in (e.get("full_name") or "").lower()
              or q in (e.get("department_name") or "").lower()
              or q in (e.get("designation") or "").lower()]
    return render_template("admin/directory.html", active="directory", people=people, q=q)


@app.route("/admin/tasks", methods=["GET", "POST"])
@admin_required
def admin_tasks():
    if request.method == "POST":
        action = request.form.get("action", "assign")
        stamp = db.now().strftime(db.DT_FMT)
        if action == "decision":
            task_id = request.form.get("task_id", type=int)
            decision = request.form.get("decision", "")
            if decision not in ("Assigned", "Rejected"):
                abort(400)
            with db.conn_ctx() as c:
                c.execute("UPDATE tasks SET status=?, assigned_by=?, updated_at=? WHERE id=? AND status='Proposed'",
                          (decision, g.admin["username"], stamp, task_id))
            flash("Employee proposal reviewed.", "success")
        else:
            emp_id = request.form.get("employee_id", type=int)
            title = request.form.get("title", "").strip()[:180]
            description = request.form.get("description", "").strip()[:4000]
            due = request.form.get("due_date", "")[:10]
            if not db.get_employee(emp_id) or not title:
                flash("Choose an employee and enter a task title.", "error")
            else:
                with db.conn_ctx() as c:
                    c.execute("""INSERT INTO tasks(employee_id,title,description,assigned_by,due_date,status,created_at,updated_at)
                        VALUES (?,?,?,?,?,'Assigned',?,?)""",
                        (emp_id, title, description, g.admin["username"], due, stamp, stamp))
                flash("Task assigned to employee.", "success")
        return redirect(url_for("admin_tasks"))
    with db.conn_ctx() as c:
        tasks = [dict(r) for r in c.execute("""SELECT t.*, e.full_name, e.email, d.name AS department_name
            FROM tasks t JOIN employees e ON e.id=t.employee_id
            LEFT JOIN departments d ON d.id=e.department_id ORDER BY
            CASE t.status WHEN 'Proposed' THEN 0 WHEN 'In Progress' THEN 1 WHEN 'Assigned' THEN 2 ELSE 3 END,
            t.created_at DESC""")]
    employees = db.list_employees(status="approved")
    return render_template("admin/tasks.html", active="tasks", tasks=tasks, employees=employees)


@app.route("/admin/tasks.xlsx")
@admin_required
def admin_tasks_xlsx():
    try:
        from openpyxl import Workbook
    except ImportError:
        abort(500, "Excel export dependency missing. Install requirements.txt.")
    with db.conn_ctx() as c:
        rows = c.execute("""SELECT t.id,e.id AS employee_id,e.full_name,d.name AS department_name,
            t.title,t.description,t.assigned_by,t.due_date,t.status,t.employee_note,t.created_at,t.updated_at
            FROM tasks t JOIN employees e ON e.id=t.employee_id LEFT JOIN departments d ON d.id=e.department_id
            ORDER BY e.full_name,t.created_at DESC""").fetchall()
    wb = Workbook(); ws = wb.active; ws.title = "Work Tracker"
    ws.append(["Task ID","Employee ID","Employee Name","Department","Task / Work","Description","Assigned By",
               "Due Date","Status","Employee Update","Created At","Last Updated"])
    for r in rows: ws.append(list(r))
    ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = min(max(max(len(str(cell.value or "")) for cell in col)+2, 12), 42)
    buf = io.BytesIO(); wb.save(buf); buf.seek(0)
    return send_file(buf, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                     as_attachment=True, download_name="employee_work_tracker.xlsx")


# =================================================================== ADMIN PANEL
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        key = f"adm:{request.remote_addr}:{username.lower()}"
        if throttled(key):
            flash("Too many failed attempts. Please wait 10 minutes.", "error")
            return render_template("admin/login.html"), 429
        row = db.get_admin_by_username(username)
        if not db.verify_password(row, request.form.get("password", "")):
            register_fail(key)
            flash("Invalid username or password.", "error")
            return render_template("admin/login.html")
        if db.admin_expired(row):
            flash("This temporary admin account has expired. Please contact the main admin.", "error")
            return render_template("admin/login.html")
        clear_fails(key)
        session.clear()
        session.permanent = True
        session["role"] = "admin"
        session["admin_id"] = row["id"]
        return redirect(url_for("admin_dashboard"))
    return render_template("admin/login.html")


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.route("/leaves", methods=["GET", "POST"])
@employee_required
def employee_leaves():
    with db.conn_ctx() as c:
        c.execute("INSERT OR IGNORE INTO leave_balances(employee_id) VALUES (?)",(g.emp["id"],))
        if request.method=="POST":
            d=request.form.get("leave_date","").strip(); typ=request.form.get("leave_type","Full Day")
            if d:
                c.execute("INSERT INTO leave_requests(employee_id,leave_date,leave_type,units,reason,created_at) VALUES (?,?,?,?,?,?)",(g.emp["id"],d,typ,0.5 if typ=="Short Leave" else 1,request.form.get("reason",""),db.now().strftime(db.DT_FMT)))
                flash("Leave request sent to admin.","success"); return redirect(url_for("employee_leaves"))
            flash("Choose a leave date.","error")
        balance=dict(c.execute("SELECT * FROM leave_balances WHERE employee_id=?",(g.emp["id"],)).fetchone())
        rows=[dict(r) for r in c.execute("SELECT * FROM leave_requests WHERE employee_id=? ORDER BY leave_date DESC",(g.emp["id"],))]
    return render_template("employee/leaves.html",active="leaves",balance=balance,rows=rows)

@app.route("/admin/leaves", methods=["GET","POST"])
@admin_required
def admin_leaves():
    with db.conn_ctx() as c:
        if request.method=="POST":
            action=request.form.get("action"); eid=int(request.form.get("employee_id",0) or 0)
            c.execute("INSERT OR IGNORE INTO leave_balances(employee_id) SELECT id FROM employees WHERE id=?",(eid,))
            if action=="balance":
                c.execute("UPDATE leave_balances SET annual_total=?,annual_used=? WHERE employee_id=?",(max(0,int(request.form.get("annual_total",30) or 30)),max(0,float(request.form.get("annual_used",0) or 0)),eid))
            elif action=="grant":
                c.execute("UPDATE leave_balances SET annual_used=MAX(0,annual_used-?) WHERE employee_id=?",(max(0,float(request.form.get("units",0) or 0)),eid))
            elif action=="review":
                rid=int(request.form.get("request_id",0) or 0); decision=request.form.get("decision")
                r=c.execute("SELECT * FROM leave_requests WHERE id=?",(rid,)).fetchone()
                if r and r["status"]=="pending" and decision in ("approved","rejected"):
                    if decision=="approved": c.execute("UPDATE leave_balances SET annual_used=annual_used+? WHERE employee_id=?",(r["units"],r["employee_id"]))
                    c.execute("UPDATE leave_requests SET status=?,reviewed_by=?,admin_note=? WHERE id=?",(decision,g.admin["username"],request.form.get("admin_note",""),rid))
            flash("Leave settings updated.","success")
        c.execute("INSERT OR IGNORE INTO leave_balances(employee_id) SELECT id FROM employees")
        employees=[dict(r) for r in c.execute("SELECT e.id,e.full_name,e.username,COALESCE(b.annual_total,30) annual_total,COALESCE(b.annual_used,0) annual_used FROM employees e LEFT JOIN leave_balances b ON b.employee_id=e.id ORDER BY e.full_name")]
        requests=[dict(r) for r in c.execute("SELECT lr.*,e.full_name,e.username FROM leave_requests lr JOIN employees e ON e.id=lr.employee_id ORDER BY CASE lr.status WHEN 'pending' THEN 0 ELSE 1 END,lr.leave_date DESC")]
    return render_template("admin/leaves.html",active="leaves",employees=employees,requests=requests)

@app.route("/admin")
@admin_required
def admin_dashboard():
    today_rows = db.attendance_report(date_from=db.today_str(), date_to=db.today_str())
    return render_template(
        "admin/dashboard.html", active="dashboard", stats=db.admin_stats(),
        pending=db.list_employees(status="pending"), today_rows=today_rows,
        absent=db.absent_today(), depts=db.departments_overview(),
        settings=db.get_settings(), req=db.required_seconds())


@app.route("/admin/approvals")
@admin_required
def admin_approvals():
    return render_template("admin/approvals.html", active="approvals",
                           pending=db.list_employees(status="pending"),
                           enrolling=db.list_employees(status="face_pending"))


def _admin_employee_action(emp_id, status, message):
    emp = db.get_employee(emp_id)
    if not emp:
        abort(404)
    if status == "rejected" and emp["status"] == "approved" and not g.admin.get("is_main"):
        flash("Only the main admin can revoke an employee's access.", "error")
    elif status == "approved" and emp["status"] not in ("pending", "rejected"):
        flash("This employee has not finished face registration yet.", "error")
    else:
        db.set_status(emp_id, status)
        flash(message.format(name=emp["full_name"]), "success")
    return redirect(request.referrer or url_for("admin_employees"))


@app.route("/admin/employee/<int:emp_id>/approve", methods=["POST"])
@admin_required
def admin_approve(emp_id):
    return _admin_employee_action(emp_id, "approved", "{name} approved. They can now login with face recognition.")


@app.route("/admin/employee/<int:emp_id>/reject", methods=["POST"])
@admin_required
def admin_reject(emp_id):
    return _admin_employee_action(emp_id, "rejected", "{name} rejected / access revoked.")


@app.route("/admin/employee/<int:emp_id>/delete", methods=["POST"])
@main_admin_required
def admin_delete_employee(emp_id):
    emp = db.get_employee(emp_id)
    if not emp:
        abort(404)
    db.delete_employee(emp_id)
    fu.delete_employee_data(emp_id, dataset_manifest())
    flash(f"{emp['full_name']} and their face data were deleted.", "success")
    return redirect(url_for("admin_employees"))


@app.route("/admin/employees")
@admin_required
def admin_employees():
    status = request.args.get("status") or None
    dept = request.args.get("department_id", type=int)
    q = request.args.get("q", "").strip() or None
    return render_template("admin/employees.html", active="employees",
                           employees=db.list_employees(status, dept, q),
                           departments=db.list_departments(), f_status=status, f_dept=dept, f_q=q or "")


@app.route("/admin/employee/<int:emp_id>")
@admin_required
def admin_employee_detail(emp_id):
    emp = db.get_employee(emp_id)
    if not emp:
        abort(404)
    month = request.args.get("month") or db.now().strftime("%Y-%m")
    n = fu.count_samples(emp_id)
    picks = sorted({i for i in (1, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n) if i >= 1}) if n else []
    return render_template("admin/employee_detail.html", active="employees", emp=emp, month=month,
                           rows=db.employee_history(emp_id, month), summary=db.month_summary(emp_id, month),
                           sample_ids=picks, sample_total=n)


@app.route("/admin/employee/<int:emp_id>/sample/<int:n>.jpg")
@admin_required
def admin_sample(emp_id, n):
    p = fu.sample_path(emp_id, n)
    if not p:
        abort(404)
    return send_file(p, mimetype="image/jpeg")


@app.route("/admin/admins")
@admin_required
def admin_admins():
    return render_template("admin/admins.html", active="admins", admins=db.list_admins(),
                           durations=ADMIN_DURATIONS)


@app.route("/admin/admins/add", methods=["POST"])
@main_admin_required
def admin_add_admin():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm", "")
    duration = request.form.get("duration", "permanent")
    err = None
    if len(username) < 3 or not username.replace("_", "").replace(".", "").isalnum():
        err = "Username must be 3+ characters (letters, numbers, . or _)."
    elif len(password) < 8:
        err = "Admin password must be at least 8 characters."
    elif password != confirm:
        err = "Passwords do not match."
    elif duration not in ADMIN_DURATIONS:
        err = "Please choose a valid access duration."
    elif db.username_taken(username):
        err = "This username is already taken."
    if err:
        flash(err, "error")
        return redirect(url_for("admin_admins"))
    days = ADMIN_DURATIONS[duration][1]
    expires = (db.now() + timedelta(days=days)).strftime(db.DT_FMT) if days else None
    if db.create_admin(username, password, expires, created_by=g.admin["username"]):
        flash(f"Admin '{username}' added" + (f" (temporary, until {db.dt_pretty(expires)})." if expires
                                             else " (permanent)."), "success")
    else:
        flash("This username is already taken.", "error")
    return redirect(url_for("admin_admins"))


@app.route("/admin/admins/<int:admin_id>/delete", methods=["POST"])
@main_admin_required
def admin_delete_admin(admin_id):
    target = db.get_admin(admin_id)
    if not target:
        abort(404)
    if target["is_main"]:
        flash("The main admin cannot be removed.", "error")
    elif db.delete_admin(admin_id):
        flash(f"Admin '{target['username']}' was removed.", "success")
    return redirect(url_for("admin_admins"))


@app.route("/admin/departments", methods=["GET", "POST"])
@admin_required
def admin_departments():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        if not name:
            flash("Department name is required.", "error")
        elif db.add_department(name, request.form.get("description", "")):
            flash("Department added.", "success")
        else:
            flash("A department with this name already exists.", "error")
        return redirect(url_for("admin_departments"))
    return render_template("admin/departments.html", active="departments", depts=db.departments_overview())


@app.route("/admin/departments/<int:dept_id>/delete", methods=["POST"])
@admin_required
def admin_delete_department(dept_id):
    if db.delete_department(dept_id):
        flash("Department deleted.", "success")
    else:
        flash("Cannot delete a department that still has employees.", "error")
    return redirect(url_for("admin_departments"))


def _report_args():
    today = db.today_str()
    return dict(
        date_from=request.args.get("from") or today,
        date_to=request.args.get("to") or today,
        department_id=request.args.get("department_id", type=int),
        employee_id=request.args.get("employee_id", type=int),
        status=request.args.get("status") or None,
    )


@app.route("/admin/attendance")
@admin_required
def admin_attendance():
    args = _report_args()
    rows = db.attendance_report(**args)
    done = [r for r in rows if r["check_out"]]
    return render_template("admin/attendance.html", active="attendance", rows=rows, a=args,
                           departments=db.list_departments(),
                           employees=db.list_employees(status="approved"),
                           total_worked=sum(r["worked"] for r in done), req=db.required_seconds(),
                           qs=request.query_string.decode())


@app.route("/admin/attendance.csv")
@admin_required
def admin_attendance_csv():
    rows = db.attendance_report(**_report_args())
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Date", "Employee", "Username", "Department", "Check In", "Check Out",
                "Worked (hh:mm)", "Required (hh:mm)", "Status", "Late"])
    for r in rows:
        w.writerow([r["work_date"], r["full_name"], r["username"], r["department_name"],
                    r["check_in_time"], r["check_out_time"], db.fmt_hm(r["worked"]).replace("h ", ":").replace("m", ""),
                    db.fmt_hm(r["required"]).replace("h ", ":").replace("m", ""), r["status"],
                    "Yes" if r["is_late"] else "No"])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=attendance_report.csv"})


@app.route("/admin/announcements", methods=["GET", "POST"])
@admin_required
def admin_announcements():
    if request.method == "POST":
        title, body = request.form.get("title", "").strip(), request.form.get("body", "").strip()
        dept = request.form.get("department_id", type=int)
        if not title or not body:
            flash("Title and message are required.", "error")
        else:
            db.add_announcement(dept, title, body)
            flash("Announcement posted.", "success")
        return redirect(url_for("admin_announcements"))
    return render_template("admin/announcements.html", active="announcements",
                           items=db.list_announcements(everything=True, limit=100),
                           departments=db.list_departments())


@app.route("/admin/announcements/<int:ann_id>/delete", methods=["POST"])
@admin_required
def admin_delete_announcement(ann_id):
    db.delete_announcement(ann_id)
    flash("Announcement deleted.", "success")
    return redirect(url_for("admin_announcements"))


@app.route("/admin/settings", methods=["GET", "POST"])
@admin_required
def admin_settings():
    if request.method == "POST":
        if request.form.get("section") == "password":
            cur, new, conf = (request.form.get(k, "") for k in ("current", "new", "confirm"))
            if not db.verify_password(g.admin, cur):
                flash("Current password is wrong.", "error")
            elif len(new) < 8:
                flash("New admin password must be at least 8 characters.", "error")
            elif new != conf:
                flash("New passwords do not match.", "error")
            else:
                db.set_admin_password(g.admin["id"], new)
                flash("Admin password updated.", "success")
        else:
            try:
                hours = float(request.form.get("required_hours", ""))
                hh, mm = request.form.get("shift_start", "09:00").split(":")
                int(hh), int(mm)
                grace = int(request.form.get("late_grace_minutes", "15"))
                if not (0.5 <= hours <= 16) or grace < 0:
                    raise ValueError
                db.save_settings(hours, f"{int(hh):02d}:{int(mm):02d}", grace)
                flash("Working-hours settings saved.", "success")
            except ValueError:
                flash("Please enter valid values (hours between 0.5 and 16).", "error")
        return redirect(url_for("admin_settings"))
    return render_template("admin/settings.html", active="settings", s=db.get_settings())


@app.route("/admin/dataset.zip")
@main_admin_required
def admin_dataset_zip():
    fu.rebuild_dataset_zip(dataset_manifest())
    return send_file(fu.ZIP_PATH, mimetype="application/zip", as_attachment=True, download_name="dataset.zip")


@app.route("/api/dataset_zip")
def dataset_zip_sync():
    """Used by sync_dataset.py:  /api/dataset_zip?key=<SYNC_KEY env var>"""
    sync_key = os.environ.get("SYNC_KEY", "")
    provided = request.args.get("key", "")
    if not ((session.get("role") == "admin" and db.get_admin(session.get("admin_id")) and db.get_admin(session.get("admin_id")).get("is_main")) or (sync_key and secrets.compare_digest(provided, sync_key))):
        return jsonify(error="Unauthorized"), 401
    fu.rebuild_dataset_zip(dataset_manifest())
    return send_file(fu.ZIP_PATH, mimetype="application/zip", as_attachment=True, download_name="dataset.zip")



# --------------------------- locations, WFH approval, Excel, messaging
@app.route("/attendance/setup", methods=["GET", "POST"])
@employee_required
def attendance_setup():
    with db.conn_ctx() as c:
        approved = c.execute("SELECT 1 FROM wfh_requests WHERE employee_id=? AND work_date=? AND status='approved'", (g.emp['id'], db.today_str())).fetchone()
    if request.method == 'POST':
        mode = request.form.get('work_mode', 'Office')
        try:
            lat, lon = float(request.form.get('latitude')), float(request.form.get('longitude'))
            if not (-90 <= lat <= 90 and -180 <= lon <= 180): raise ValueError
        except (TypeError, ValueError):
            flash('Please allow location access to record your check-in location.', 'error')
            return redirect(url_for('attendance_setup'))
        location = 'Work From Home' if mode == 'Home' else ('Office / On-site' if mode == 'Office' else 'Other location')
        session['attendance_choice'] = {'mode': mode, 'location': location, 'latitude': lat, 'longitude': lon}
        return redirect(url_for('scan', action='checkin'))
    return render_template('employee/attendance_setup.html', active='dashboard', locations=[], approved=bool(approved), assigned='')

@app.route('/wfh-request', methods=['GET','POST'])
@employee_required
def wfh_request():
    if request.method == 'POST':
        reason=request.form.get('reason','').strip()
        if not reason:
            flash('Please provide a reason for your work-from-home request.', 'error')
            return redirect(url_for('wfh_request'))
        with db.conn_ctx() as c:
            c.execute("INSERT INTO wfh_requests(employee_id,work_date,reason,status,created_at) VALUES (?,?,?,'pending',?) ON CONFLICT(employee_id,work_date) DO UPDATE SET reason=excluded.reason,status='pending',created_at=excluded.created_at", (g.emp['id'],db.today_str(),reason,db.now().strftime(db.DT_FMT)))
        flash('WFH request sent to admin for approval.', 'success'); return redirect(url_for('wfh_request'))
    with db.conn_ctx() as c:
        rows=[dict(x) for x in c.execute('SELECT * FROM wfh_requests WHERE employee_id=? ORDER BY work_date DESC LIMIT 30',(g.emp['id'],))]
    return render_template('employee/wfh_request.html',active='wfh',rows=rows)

@app.route('/admin/locations', methods=['GET','POST'])
@admin_required
def admin_locations():
    if request.method == 'POST':
        action = request.form.get('form_action', '').strip()
        # Only the main admin may configure office locations or employee assignments.
        if action in ('add_location', 'update_coordinates', 'assign_employee') and not g.admin.get('is_main'):
            abort(403, 'Only the main admin can change office locations.')
        with db.conn_ctx() as c:
            if action == 'add_location':
                name = request.form.get('name', '').strip()
                try:
                    lat = float(request.form.get('latitude', ''))
                    lon = float(request.form.get('longitude', ''))
                    radius = int(request.form.get('radius_m', 150))
                    if not name or not (-90 <= lat <= 90) or not (-180 <= lon <= 180) or not (25 <= radius <= 2000):
                        raise ValueError
                except (TypeError, ValueError):
                    flash('Enter a location name, valid latitude/longitude, and a radius from 25 to 2000 meters.', 'error')
                    return redirect(url_for('admin_locations'))
                try:
                    c.execute('INSERT INTO locations(name,active,latitude,longitude,radius_m) VALUES (?,1,?,?,?)', (name, lat, lon, radius))
                    flash('Office location added and geofence configured.', 'success')
                except Exception:
                    flash('That location already exists or could not be added.', 'error')
            elif action == 'update_coordinates':
                loc_id = request.form.get('location_id', type=int)
                try:
                    lat = float(request.form.get('latitude', ''))
                    lon = float(request.form.get('longitude', ''))
                    radius = int(request.form.get('radius_m', 150))
                    if not loc_id or not (-90 <= lat <= 90) or not (-180 <= lon <= 180) or not (25 <= radius <= 2000):
                        raise ValueError
                except (TypeError, ValueError):
                    flash('Choose an office and enter valid coordinates and a radius from 25 to 2000 meters.', 'error')
                    return redirect(url_for('admin_locations'))
                cur = c.execute('UPDATE locations SET latitude=?,longitude=?,radius_m=? WHERE id=?', (lat, lon, radius, loc_id))
                flash('Office coordinates saved successfully.' if cur.rowcount else 'Office location was not found.', 'success' if cur.rowcount else 'error')
            elif action == 'assign_employee':
                emp_id = request.form.get('employee_id', type=int)
                assigned = request.form.get('assigned_location', '').strip()
                emp = c.execute("SELECT id FROM employees WHERE id=? AND status='approved'", (emp_id,)).fetchone() if emp_id else None
                active_location = c.execute('SELECT name FROM locations WHERE name=? AND active=1', (assigned,)).fetchone() if assigned else None
                if not emp:
                    flash('Please select a valid approved employee.', 'error')
                elif assigned and not active_location:
                    flash('Please assign an active office location.', 'error')
                else:
                    c.execute('UPDATE employees SET assigned_location=? WHERE id=?', (assigned, emp_id))
                    flash('Employee office assignment saved.', 'success')
            else:
                flash('Unknown form action. Please refresh the page and try again.', 'error')
        return redirect(url_for('admin_locations'))
    with db.conn_ctx() as c:
        locations = [dict(x) for x in c.execute('SELECT * FROM locations ORDER BY name')]
        employees = [dict(x) for x in c.execute("SELECT id,full_name,assigned_location FROM employees WHERE status='approved' ORDER BY full_name")]
        requests = [dict(x) for x in c.execute("SELECT w.*,e.full_name,e.username FROM wfh_requests w JOIN employees e ON e.id=w.employee_id WHERE w.status='pending' ORDER BY w.created_at")]
    return render_template('admin/locations.html', active='locations', locations=locations, employees=employees, requests=requests)

@app.route('/admin/location/<int:loc_id>/toggle',methods=['POST'])
@main_admin_required
def admin_location_toggle(loc_id):
    with db.conn_ctx() as c: c.execute('UPDATE locations SET active=CASE active WHEN 1 THEN 0 ELSE 1 END WHERE id=?',(loc_id,))
    return redirect(url_for('admin_locations'))

@app.route('/admin/wfh/<int:req_id>/<decision>',methods=['POST'])
@admin_required
def admin_wfh_decide(req_id,decision):
    if decision not in ('approve','reject'): abort(404)
    with db.conn_ctx() as c: c.execute('UPDATE wfh_requests SET status=?,reviewed_by=? WHERE id=? AND status=\'pending\'',('approved' if decision=='approve' else 'rejected',g.admin['username'],req_id))
    flash('WFH request updated.','success'); return redirect(url_for('admin_locations'))

@app.route('/admin/attendance.xlsx')
@admin_required
def admin_attendance_xlsx():
    try: from openpyxl import Workbook
    except ImportError: abort(500,'Excel export dependency missing. Install requirements.txt.')
    rows=db.attendance_report(**_report_args()); buf=io.BytesIO(); wb=Workbook(); ws=wb.active; ws.title='Attendance'
    ws.append(['Employee ID','Name','Department','Date','Check-in time','Check-out time','Work mode','Location','Latitude','Longitude','Worked','Status'])
    with db.conn_ctx() as c:
        for r in rows:
            extra=c.execute('SELECT work_mode,location_name,latitude,longitude FROM attendance WHERE employee_id=? AND work_date=?',(r['employee_id'],r['work_date'])).fetchone()
            ws.append([r['employee_id'],r['full_name'],r['department_name'],r['work_date'],r['check_in_time'],r['check_out_time'],extra['work_mode'] if extra else 'Office',extra['location_name'] if extra else '',extra['latitude'] if extra else '',extra['longitude'] if extra else '',db.fmt_hm(r['worked']),r['status']])
    ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width=min(max(max(len(str(cell.value or '')) for cell in col)+2,12),32)
    wb.save(buf); buf.seek(0)
    return send_file(buf,mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',as_attachment=True,download_name='attendance_report.xlsx')

@app.route('/messages',methods=['GET','POST'])
@employee_required
def employee_messages():
    if request.method=='POST':
        subject=request.form.get('subject','').strip(); body=request.form.get('body','').strip(); role=request.form.get('recipient_role','HR')
        if role not in ('HR','Admin','CEO','All (CEO, Admin & HR)'): role='HR'
        if subject and body:
            with db.conn_ctx() as c: c.execute('INSERT INTO messages(employee_id,recipient_role,subject,body,created_at) VALUES (?,?,?,?,?)',(g.emp['id'],role,subject,body,db.now().strftime(db.DT_FMT)))
            flash('Message sent.','success')
        else: flash('Subject and message are required.','error')
        return redirect(url_for('employee_messages'))
    with db.conn_ctx() as c: rows=[dict(x) for x in c.execute('SELECT * FROM messages WHERE employee_id=? ORDER BY created_at DESC',(g.emp['id'],))]
    return render_template('employee/messages.html',active='messages',rows=rows)

@app.route('/admin/messages',methods=['GET','POST'])
@admin_required
def admin_messages():
    if request.method=='POST':
        mid=request.form.get('message_id',type=int); reply=request.form.get('reply','').strip()
        if mid and reply:
            with db.conn_ctx() as c: c.execute("UPDATE messages SET reply=?,status='replied',replied_at=? WHERE id=?",(reply,db.now().strftime(db.DT_FMT),mid))
        return redirect(url_for('admin_messages'))
    with db.conn_ctx() as c: rows=[dict(x) for x in c.execute('SELECT m.*,e.full_name,e.username FROM messages m JOIN employees e ON e.id=m.employee_id ORDER BY m.created_at DESC')]
    return render_template('admin/messages.html',active='messages',rows=rows)


# ------------------------------------------------------------------ errors
@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
def http_error(e):
    return render_template("error.html", code=e.code, message=getattr(e, "description", "")), e.code


if __name__ == "__main__":
    ssl_ctx = "adhoc" if os.environ.get("HTTPS") == "1" else None   # needs: pip install cryptography
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True,
            debug=os.environ.get("DEBUG") == "1", ssl_context=ssl_ctx)
