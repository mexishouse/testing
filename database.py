"""
database.py
SQLite layer for the Face Recognition Attendance System.

Tables
  admins          - admin accounts (password hashed). One MAIN admin (muzammil) + extra admins,
                    which can be permanent or temporary (expires_at)
  departments     - departments
  employees       - employee accounts + face-enrolment / approval status
  attendance      - one row per employee per day (check_in / check_out / worked_seconds)
  announcements   - notices (for one department, or for everyone)
  settings        - required hours, shift start, late grace
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", BASE_DIR)          # Railway volume support
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "attendance.db")

DEFAULT_ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "muzammil")
DEFAULT_ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "muzammil182921")

DT_FMT = "%Y-%m-%d %H:%M:%S"
D_FMT = "%Y-%m-%d"
MIN_SESSION_SECONDS = 60          # cannot check out within 1 minute of checking in

try:
    TZ = ZoneInfo(os.environ.get("APP_TIMEZONE", "Asia/Karachi"))
except Exception:                 # tzdata missing -> fall back to server time
    TZ = None

DEFAULT_DEPARTMENTS = [
    ("IT Software", "Software development, QA and IT systems"),
    ("Human Resources", "Hiring, payroll and employee affairs"),
    ("Finance", "Accounts, budgets and audits"),
    ("Marketing", "Branding, campaigns and social media"),
    ("Sales", "Clients, leads and revenue"),
    ("Operations", "Day-to-day business operations"),
    ("Administration", "Office administration and facilities"),
    ("Production", "Manufacturing and production"),
    ("Quality Assurance", "Quality control and testing"),
    ("Research & Development", "New products and research"),
    ("Customer Support", "Helpdesk and customer care"),
    ("Logistics & Supply Chain", "Warehouse, shipping and procurement"),
]

DEFAULT_SETTINGS = {
    "required_hours": "8",
    "shift_start": "09:00",
    "late_grace_minutes": "15",
}


def now():
    """Current local datetime (naive) in the configured timezone."""
    return datetime.now(TZ).replace(tzinfo=None) if TZ else datetime.now()


def today_str():
    return now().strftime(D_FMT)


@contextmanager
def conn_ctx():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _rows(cur):
    return [dict(r) for r in cur.fetchall()]


def _row(cur):
    r = cur.fetchone()
    return dict(r) if r else None


# ---------------------------------------------------------------- init
def init_db():
    with conn_ctx() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_default_password INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            is_main INTEGER DEFAULT 0,          -- 1 = main admin (only one)
            expires_at TEXT,                    -- NULL = permanent, otherwise temporary admin
            created_by TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT DEFAULT '',
            designation TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            profile_picture TEXT DEFAULT '',
            social_linkedin TEXT DEFAULT '',
            social_facebook TEXT DEFAULT '',
            social_instagram TEXT DEFAULT '',
            social_x TEXT DEFAULT '',
            social_website TEXT DEFAULT '',
            department_id INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'face_pending',
                -- face_pending -> pending -> approved | rejected
            samples_captured INTEGER DEFAULT 0,
            flag_note TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            approved_at TEXT,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            work_date TEXT NOT NULL,
            check_in TEXT NOT NULL,
            check_out TEXT,
            worked_seconds INTEGER,
            is_late INTEGER DEFAULT 0,
            UNIQUE (employee_id, work_date),
            FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS announcements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER,            -- NULL = everyone
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS locations (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL, active INTEGER DEFAULT 1);
        CREATE TABLE IF NOT EXISTS wfh_requests (id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL, work_date TEXT NOT NULL, reason TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL, reviewed_by TEXT DEFAULT '', FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE, UNIQUE(employee_id, work_date));
        CREATE TABLE IF NOT EXISTS leave_balances (
            employee_id INTEGER PRIMARY KEY, annual_total INTEGER NOT NULL DEFAULT 30,
            annual_used REAL NOT NULL DEFAULT 0, short_used REAL NOT NULL DEFAULT 0,
            FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS leave_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL,
            leave_date TEXT NOT NULL, leave_type TEXT NOT NULL DEFAULT 'Full Day',
            units REAL NOT NULL DEFAULT 1, reason TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL, reviewed_by TEXT DEFAULT '', admin_note TEXT DEFAULT '',
            FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL, recipient_role TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL, status TEXT DEFAULT 'open', reply TEXT DEFAULT '', created_at TEXT NOT NULL, replied_at TEXT, FOREIGN KEY(employee_id) REFERENCES employees(id) ON DELETE CASCADE);
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            assigned_by TEXT DEFAULT '',
            due_date TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'Assigned',
            employee_note TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_task_employee ON tasks(employee_id, status);
        CREATE INDEX IF NOT EXISTS idx_att_date ON attendance(work_date);
        CREATE INDEX IF NOT EXISTS idx_emp_dept ON employees(department_id);
        """)

        c.execute("INSERT OR IGNORE INTO leave_balances(employee_id) SELECT id FROM employees")
        # --- migrate databases created before roles existed
        cols = {r[1] for r in c.execute("PRAGMA table_info(admins)")}
        for name, ddl in (("is_main", "INTEGER DEFAULT 0"), ("expires_at", "TEXT"),
                          ("created_by", "TEXT DEFAULT ''")):
            if name not in cols:
                c.execute(f"ALTER TABLE admins ADD COLUMN {name} {ddl}")
        # Safe additive migrations for attendance and employee location/work mode.
        att_cols = {r[1] for r in c.execute("PRAGMA table_info(attendance)")}
        for name, ddl in (("work_mode", "TEXT DEFAULT 'Office'"), ("location_name", "TEXT DEFAULT ''"), ("latitude", "REAL"), ("longitude", "REAL")):
            if name not in att_cols: c.execute(f"ALTER TABLE attendance ADD COLUMN {name} {ddl}")
        loc_cols = {r[1] for r in c.execute("PRAGMA table_info(locations)")}
        for name, ddl in (("latitude", "REAL"), ("longitude", "REAL"), ("radius_m", "INTEGER DEFAULT 150")):
            if name not in loc_cols: c.execute(f"ALTER TABLE locations ADD COLUMN {name} {ddl}")
        emp_cols = {r[1] for r in c.execute("PRAGMA table_info(employees)")}
        for name, ddl in (("phone", "TEXT DEFAULT ''"), ("profile_picture", "TEXT DEFAULT ''"),
                          ("social_linkedin", "TEXT DEFAULT ''"), ("social_facebook", "TEXT DEFAULT ''"),
                          ("social_instagram", "TEXT DEFAULT ''"), ("social_x", "TEXT DEFAULT ''"),
                          ("social_website", "TEXT DEFAULT ''")):
            if name not in emp_cols: c.execute(f"ALTER TABLE employees ADD COLUMN {name} {ddl}")
        if "assigned_location" not in emp_cols: c.execute("ALTER TABLE employees ADD COLUMN assigned_location TEXT DEFAULT ''")
        for loc in ("Johar Town", "Superior Gold Campus"):
            c.execute("INSERT OR IGNORE INTO locations(name, active) VALUES (?,1)", (loc,))
        for k, v in DEFAULT_SETTINGS.items():
            c.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (k, v))
        if c.execute("SELECT COUNT(*) FROM departments").fetchone()[0] == 0:
            c.executemany("INSERT INTO departments(name, description) VALUES (?, ?)", DEFAULT_DEPARTMENTS)
        if c.execute("SELECT COUNT(*) FROM admins").fetchone()[0] == 0:
            c.execute(
                "INSERT INTO admins(username, password_hash, is_default_password, created_at, is_main) "
                "VALUES (?,?,1,?,1)",
                (DEFAULT_ADMIN_USERNAME, generate_password_hash(DEFAULT_ADMIN_PASSWORD), now().strftime(DT_FMT)),
            )
        # exactly one main admin: the configured one (muzammil). Never temporary.
        if c.execute("SELECT COUNT(*) FROM admins WHERE is_main = 1").fetchone()[0] != 1:
            c.execute("UPDATE admins SET is_main = 0")
            c.execute("UPDATE admins SET is_main = 1, expires_at = NULL WHERE lower(username) = lower(?)",
                      (DEFAULT_ADMIN_USERNAME,))
            if c.execute("SELECT COUNT(*) FROM admins WHERE is_main = 1").fetchone()[0] == 0:
                c.execute("UPDATE admins SET is_main = 1, expires_at = NULL "
                          "WHERE id = (SELECT MIN(id) FROM admins)")


# ---------------------------------------------------------------- settings
def get_settings():
    with conn_ctx() as c:
        d = {r["key"]: r["value"] for r in c.execute("SELECT * FROM settings")}
    out = dict(DEFAULT_SETTINGS)
    out.update(d)
    return out


def required_seconds(settings=None):
    s = settings or get_settings()
    try:
        return int(float(s["required_hours"]) * 3600)
    except ValueError:
        return 8 * 3600


def save_settings(required_hours, shift_start, late_grace_minutes):
    with conn_ctx() as c:
        for k, v in (("required_hours", required_hours), ("shift_start", shift_start),
                     ("late_grace_minutes", late_grace_minutes)):
            c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)", (k, str(v)))


# ---------------------------------------------------------------- admins
def get_admin_by_username(username):
    with conn_ctx() as c:
        return _row(c.execute("SELECT * FROM admins WHERE username = ?", (username,)))


def get_admin(admin_id):
    with conn_ctx() as c:
        return _row(c.execute("SELECT * FROM admins WHERE id = ?", (admin_id,)))


def set_admin_password(admin_id, new_password):
    with conn_ctx() as c:
        c.execute("UPDATE admins SET password_hash = ?, is_default_password = 0 WHERE id = ?",
                  (generate_password_hash(new_password), admin_id))


def verify_password(row, password):
    return bool(row) and check_password_hash(row["password_hash"], password)


def admin_expired(row):
    """True when a temporary admin's access time is over (the main admin never expires)."""
    if not row or row.get("is_main") or not row.get("expires_at"):
        return False
    try:
        return now() >= _parse(row["expires_at"])
    except (ValueError, TypeError):
        return True            # unreadable date -> fail closed


def list_admins():
    with conn_ctx() as c:
        rows = _rows(c.execute("SELECT * FROM admins ORDER BY is_main DESC, created_at"))
    for r in rows:
        r["expired"] = admin_expired(r)
    return rows


def create_admin(username, password, expires_at=None, created_by=""):
    """Add a normal (never main) admin. expires_at = None -> permanent, else 'YYYY-MM-DD HH:MM:SS'."""
    try:
        with conn_ctx() as c:
            c.execute(
                "INSERT INTO admins(username, password_hash, is_default_password, created_at, "
                "is_main, expires_at, created_by) VALUES (?,?,1,?,0,?,?)",
                (username.strip(), generate_password_hash(password), now().strftime(DT_FMT),
                 expires_at, created_by))
        return True
    except sqlite3.IntegrityError:
        return False


def delete_admin(admin_id):
    """Removes a non-main admin. The main admin can never be deleted."""
    with conn_ctx() as c:
        cur = c.execute("DELETE FROM admins WHERE id = ? AND is_main = 0", (admin_id,))
        return cur.rowcount > 0


# ---------------------------------------------------------------- departments
def list_departments():
    with conn_ctx() as c:
        return _rows(c.execute("SELECT * FROM departments ORDER BY name"))


def get_department(dept_id):
    with conn_ctx() as c:
        return _row(c.execute("SELECT * FROM departments WHERE id = ?", (dept_id,)))


def add_department(name, description=""):
    try:
        with conn_ctx() as c:
            c.execute("INSERT INTO departments(name, description) VALUES (?, ?)",
                      (name.strip(), description.strip()))
        return True
    except sqlite3.IntegrityError:
        return False


def delete_department(dept_id):
    """Only allowed when no employee belongs to it."""
    with conn_ctx() as c:
        n = c.execute("SELECT COUNT(*) FROM employees WHERE department_id = ?", (dept_id,)).fetchone()[0]
        if n:
            return False
        c.execute("DELETE FROM departments WHERE id = ?", (dept_id,))
    return True


def departments_overview():
    """Every department with employee counts and today's presence."""
    today = today_str()
    with conn_ctx() as c:
        return _rows(c.execute("""
            SELECT d.id, d.name, d.description,
                   COUNT(DISTINCT CASE WHEN e.status = 'approved' THEN e.id END) AS approved_count,
                   COUNT(DISTINCT e.id) AS total_count,
                   COUNT(DISTINCT a.id) AS present_today
            FROM departments d
            LEFT JOIN employees e ON e.department_id = d.id
            LEFT JOIN attendance a ON a.employee_id = e.id AND a.work_date = ?
            GROUP BY d.id ORDER BY d.name
        """, (today,)))


# ---------------------------------------------------------------- employees
def username_taken(username):
    with conn_ctx() as c:
        a = c.execute("SELECT 1 FROM employees WHERE lower(username) = lower(?)", (username,)).fetchone()
        b = c.execute("SELECT 1 FROM admins WHERE lower(username) = lower(?)", (username,)).fetchone()
    return bool(a or b)


def create_employee(full_name, username, password, department_id, email="", designation=""):
    with conn_ctx() as c:
        cur = c.execute("""
            INSERT INTO employees(full_name, username, password_hash, email, designation,
                                  department_id, status, created_at)
            VALUES (?,?,?,?,?,?, 'face_pending', ?)
        """, (full_name.strip(), username.strip(), generate_password_hash(password),
              email.strip(), designation.strip(), department_id, now().strftime(DT_FMT)))
        return cur.lastrowid


_EMP_SELECT = """
    SELECT e.*, d.name AS department_name
    FROM employees e JOIN departments d ON d.id = e.department_id
"""


def get_employee(emp_id):
    with conn_ctx() as c:
        return _row(c.execute(_EMP_SELECT + " WHERE e.id = ?", (emp_id,)))


def get_employee_by_username(username):
    with conn_ctx() as c:
        return _row(c.execute(_EMP_SELECT + " WHERE lower(e.username) = lower(?)", (username,)))


def list_employees(status=None, department_id=None, search=None):
    q, p = _EMP_SELECT + " WHERE 1=1", []
    if status:
        q += " AND e.status = ?"
        p.append(status)
    if department_id:
        q += " AND e.department_id = ?"
        p.append(department_id)
    if search:
        q += " AND (e.full_name LIKE ? OR e.username LIKE ?)"
        p += [f"%{search}%", f"%{search}%"]
    q += " ORDER BY e.full_name"
    with conn_ctx() as c:
        return _rows(c.execute(q, p))


def set_status(emp_id, status, flag_note=None):
    with conn_ctx() as c:
        if status == "approved":
            c.execute("UPDATE employees SET status=?, approved_at=? WHERE id=?",
                      (status, now().strftime(DT_FMT), emp_id))
        else:
            c.execute("UPDATE employees SET status=? WHERE id=?", (status, emp_id))
        if flag_note is not None:
            c.execute("UPDATE employees SET flag_note=? WHERE id=?", (flag_note, emp_id))


def set_samples(emp_id, count):
    with conn_ctx() as c:
        c.execute("UPDATE employees SET samples_captured=? WHERE id=?", (count, emp_id))


def set_employee_password(emp_id, new_password):
    with conn_ctx() as c:
        c.execute("UPDATE employees SET password_hash=? WHERE id=?",
                  (generate_password_hash(new_password), emp_id))


def delete_employee(emp_id):
    with conn_ctx() as c:
        c.execute("DELETE FROM attendance WHERE employee_id=?", (emp_id,))
        c.execute("DELETE FROM employees WHERE id=?", (emp_id,))


def department_members(dept_id):
    """Approved members of ONE department + today's attendance state (used by employee panel)."""
    today = today_str()
    with conn_ctx() as c:
        return _rows(c.execute("""
            SELECT e.id, e.full_name, e.designation, a.check_in, a.check_out
            FROM employees e
            LEFT JOIN attendance a ON a.employee_id = e.id AND a.work_date = ?
            WHERE e.department_id = ? AND e.status = 'approved'
            ORDER BY e.full_name
        """, (today, dept_id)))


# ---------------------------------------------------------------- attendance
def _parse(dt):
    return datetime.strptime(dt, DT_FMT) if dt else None


def enrich(row, req_seconds, now_dt=None):
    """Add worked time, status and status colour to an attendance row."""
    now_dt = now_dt or now()
    ci, co = _parse(row["check_in"]), _parse(row["check_out"])
    if co:
        worked = row["worked_seconds"] if row["worked_seconds"] is not None else int((co - ci).total_seconds())
        if worked >= req_seconds:
            status, cls = "Completed", "ok"
        else:
            status, cls = "Short Hours", "warn"
    elif row["work_date"] == now_dt.strftime(D_FMT):
        worked = max(0, int((now_dt - ci).total_seconds()))
        status, cls = "Working", "live"
    else:
        worked = 0
        status, cls = "No Check-out", "bad"
    row = dict(row)
    row.update(
        worked=worked,
        required=req_seconds,
        short_by=max(0, req_seconds - worked) if co else None,
        overtime=max(0, worked - req_seconds) if co else None,
        status=status,
        status_class=cls,
        check_in_time=ci.strftime("%I:%M:%S %p"),
        check_out_time=co.strftime("%I:%M:%S %p") if co else "—",
    )
    return row


def get_today_record(emp_id):
    with conn_ctx() as c:
        return _row(c.execute("SELECT * FROM attendance WHERE employee_id=? AND work_date=?",
                              (emp_id, today_str())))


def check_in(emp_id):
    """Returns (ok, message)."""
    n = now()
    settings = get_settings()
    with conn_ctx() as c:
        if c.execute("SELECT 1 FROM attendance WHERE employee_id=? AND work_date=?",
                     (emp_id, n.strftime(D_FMT))).fetchone():
            return False, "You have already checked in today."
        h, m = [int(x) for x in settings["shift_start"].split(":")]
        grace = int(settings["late_grace_minutes"])
        late_after = n.replace(hour=h, minute=m, second=0, microsecond=0) + timedelta(minutes=grace)
        is_late = 1 if n > late_after else 0
        c.execute("INSERT INTO attendance(employee_id, work_date, check_in, is_late) VALUES (?,?,?,?)",
                  (emp_id, n.strftime(D_FMT), n.strftime(DT_FMT), is_late))
    return True, "Checked in at " + n.strftime("%I:%M %p") + (" (late)." if is_late else ".")


def check_out(emp_id, allow_early=False):
    """Returns (ok, message). Early leave only when allow_early is True."""
    n = now()
    req = required_seconds()
    with conn_ctx() as c:
        rec = _row(c.execute("SELECT * FROM attendance WHERE employee_id=? AND work_date=?",
                             (emp_id, n.strftime(D_FMT))))
        if not rec:
            return False, "You have not checked in today."
        if rec["check_out"]:
            return False, "You have already checked out today."
        worked = int((n - _parse(rec["check_in"])).total_seconds())
        if worked < MIN_SESSION_SECONDS:
            return False, "Please wait at least a minute after checking in."
        if worked < req and not allow_early:
            return False, "Required working hours are not complete yet."
        c.execute("UPDATE attendance SET check_out=?, worked_seconds=? WHERE id=?",
                  (n.strftime(DT_FMT), worked, rec["id"]))
    return True, "Checked out at " + n.strftime("%I:%M %p") + ". Worked " + fmt_hm(worked) + "."


def dt_pretty(value):
    try:
        return _parse(value).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return value or "—"


def fmt_hm(seconds):
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h, m = seconds // 3600, (seconds % 3600) // 60
    return f"{h}h {m:02d}m"


def attendance_report(date_from=None, date_to=None, department_id=None, employee_id=None, status=None):
    q = """
        SELECT a.*, e.full_name, e.username, e.department_id, d.name AS department_name
        FROM attendance a
        JOIN employees e ON e.id = a.employee_id
        JOIN departments d ON d.id = e.department_id
        WHERE 1=1
    """
    p = []
    if date_from:
        q += " AND a.work_date >= ?"; p.append(date_from)
    if date_to:
        q += " AND a.work_date <= ?"; p.append(date_to)
    if department_id:
        q += " AND e.department_id = ?"; p.append(department_id)
    if employee_id:
        q += " AND a.employee_id = ?"; p.append(employee_id)
    q += " ORDER BY a.work_date DESC, a.check_in DESC"
    with conn_ctx() as c:
        rows = _rows(c.execute(q, p))
    req = required_seconds()
    nw = now()
    rows = [enrich(r, req, nw) for r in rows]
    if status:
        rows = [r for r in rows if r["status"] == status]
    return rows


def employee_history(emp_id, month=None, limit=None):
    """month = 'YYYY-MM' or None for latest records."""
    q = "SELECT * FROM attendance WHERE employee_id = ?"
    p = [emp_id]
    if month:
        q += " AND work_date LIKE ?"; p.append(month + "-%")
    q += " ORDER BY work_date DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    with conn_ctx() as c:
        rows = _rows(c.execute(q, p))
    req = required_seconds()
    nw = now()
    return [enrich(r, req, nw) for r in rows]


def month_summary(emp_id, month=None):
    month = month or now().strftime("%Y-%m")
    rows = employee_history(emp_id, month)
    done = [r for r in rows if r["check_out"]]
    total = sum(r["worked"] for r in done)
    return {
        "month": month,
        "days_present": len(rows),
        "total_seconds": total,
        "avg_seconds": int(total / len(done)) if done else 0,
        "late_days": sum(1 for r in rows if r["is_late"]),
        "short_days": sum(1 for r in rows if r["status"] == "Short Hours"),
        "missed_checkouts": sum(1 for r in rows if r["status"] == "No Check-out"),
    }


# ---------------------------------------------------------------- admin stats
def admin_stats():
    today = today_str()
    with conn_ctx() as c:
        one = lambda q, p=(): c.execute(q, p).fetchone()[0]
        approved = one("SELECT COUNT(*) FROM employees WHERE status='approved'")
        present = one("SELECT COUNT(*) FROM attendance WHERE work_date=?", (today,))
        working = one("SELECT COUNT(*) FROM attendance WHERE work_date=? AND check_out IS NULL", (today,))
        return {
            "approved": approved,
            "pending": one("SELECT COUNT(*) FROM employees WHERE status='pending'"),
            "enrolling": one("SELECT COUNT(*) FROM employees WHERE status='face_pending'"),
            "departments": one("SELECT COUNT(*) FROM departments"),
            "present": present,
            "working": working,
            "left": present - working,
            "absent": max(0, approved - present),
            "late": one("SELECT COUNT(*) FROM attendance WHERE work_date=? AND is_late=1", (today,)),
        }


def absent_today():
    today = today_str()
    with conn_ctx() as c:
        return _rows(c.execute("""
            SELECT e.id, e.full_name, d.name AS department_name
            FROM employees e JOIN departments d ON d.id = e.department_id
            WHERE e.status='approved'
              AND NOT EXISTS (SELECT 1 FROM attendance a WHERE a.employee_id=e.id AND a.work_date=?)
            ORDER BY d.name, e.full_name
        """, (today,)))


# ---------------------------------------------------------------- announcements
def add_announcement(department_id, title, body):
    with conn_ctx() as c:
        c.execute("INSERT INTO announcements(department_id, title, body, created_at) VALUES (?,?,?,?)",
                  (department_id, title.strip(), body.strip(), now().strftime(DT_FMT)))


def delete_announcement(ann_id):
    with conn_ctx() as c:
        c.execute("DELETE FROM announcements WHERE id=?", (ann_id,))


def list_announcements(department_id=None, everything=False, limit=20):
    """Employee panel: only 'everyone' notices + the employee's own department."""
    q = """SELECT a.*, d.name AS department_name FROM announcements a
           LEFT JOIN departments d ON d.id = a.department_id"""
    p = []
    if not everything:
        q += " WHERE a.department_id IS NULL OR a.department_id = ?"
        p.append(department_id)
    q += f" ORDER BY a.created_at DESC LIMIT {int(limit)}"
    with conn_ctx() as c:
        return _rows(c.execute(q, p))


def count_status(status):
    with conn_ctx() as c:
        return c.execute("SELECT COUNT(*) FROM employees WHERE status=?", (status,)).fetchone()[0]
