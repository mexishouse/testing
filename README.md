# CyberAttend v2 — Face Recognition Attendance (Admin Panel + Employee Panel)

Do bilkul alag panels:

| Panel | URL | Login |
|---|---|---|
| **Admin** | `/admin/login` | username + password |
| **Employee** | `/login` | username + password **+ face recognition** |

## Employee ka poora flow

1. `/register` — naam, department, username, password.
2. **Face registration** — camera automatically 50 samples leta hai. Har sample `dataset/<id>/` mein save hota hai
   aur saath hi **`dataset_archive/dataset.zip`** mein pack hota rehta hai.
3. **"Admin's approval pending"** screen — page khud update hota hai.
4. Admin **Approvals** page par face samples dekh kar *Approve* karta hai.
5. Employee `/login` → password → **face scan** → Dashboard (face match na ho to login nahi hota).
6. Dashboard par **Check In (face scan)** → working timer chalta hai.
7. Kaam khatam karke **Check Out (face scan)** → working hours save. Agar required hours poore nahi hue to
   warning aati hai aur din **"Short Hours"** record hota hai (admin ko nazar aata hai).

## Department-wise access
Employee sirf **apni department** ki information dekh sakta hai (team members, department notices, attendance).
Doosri department ka URL kholne par `403` aata hai. IT Software ka employee sirf IT Software dekhta hai; baqi sab kuch sirf admin dekhta hai.

## Admin panel
Dashboard (aaj ki attendance + working hours, absent list, department chart) · Approvals (face samples ke saath) ·
Employees (approve / revoke / delete) · Admins · Attendance reports (date, department, employee, status filter + **CSV**) ·
Departments · Announcements (sab ko ya sirf ek department ko) · Settings (required hours, shift time, late grace, password) ·
**Dataset ZIP** download.

## Admin roles (naya)
| Role | Kya kar sakta hai |
|---|---|
| **Main admin (muzammil)** | Sab kuch. **Sirf wohi** employees ko revoke/delete kar sakta hai, aur admins add/remove kar sakta hai. |
| **Admin (permanent)** | Approvals, attendance, departments, announcements, settings. **Kisi ko remove nahi kar sakta.** |
| **Temporary admin** | Permanent admin jaisa, lekin muqarrar waqt (1 / 3 / 7 / 30 din) ke baad access **khud band** ho jata hai. |

Admins page: `/admin/admins`. Main admin ko kabhi delete nahi kiya ja sakta. Purani `attendance.db` startup par khud migrate ho jati hai
(muzammil main admin ban jata hai). Pending registration ko *Reject* karna har admin kar sakta hai; approved employee ko *Revoke* ya *Delete* sirf main admin.

## Chalane ka tareeqa
```bash
python -m venv venv
venv\Scripts\activate            # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Browser: `http://127.0.0.1:5000` (employee) · `http://127.0.0.1:5000/admin/login` (admin)

Default admin: **muzammil / muzammil182921** — pehli dafa login ke baad **Settings → Change password** zaroor karein.

> **Camera:** browser camera sirf **HTTPS** ya **localhost** par chalta hai. Phone se LAN par test karna ho to
> `pip install cryptography` karke `HTTPS=1 python app.py` chalayen (self-signed certificate, browser warning accept karein).
> Railway par HTTPS pehle se hota hai.

## Settings / Environment variables
| Variable | Matlab | Default |
|---|---|---|
| `DATA_DIR` | database + dataset + trainer ka folder (Railway Volume path) | project folder |
| `SECRET_KEY` | session key (na ho to `secret.key` file auto-ban jati hai) | auto |
| `APP_TIMEZONE` | attendance ka time zone | `Asia/Karachi` |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | pehli dafa admin banane ke liye | muzammil / muzammil182921 |
| `CONFIDENCE_THRESHOLD` | face match strictness (chota = strict) | `75` |
| `STREAK_NEEDED` | login/check-in ke liye lagatar kitne frames match hon | `3` |
| `SAMPLES_NEEDED` | registration par face samples | `50` |
| `SYNC_KEY` | `sync_dataset.py` ke liye | — |
| `COOKIE_SECURE=1` | HTTPS par secure cookies | off |

**Railway:** Volume mount karein aur `DATA_DIR` us path par set karein, warna redeploy par database/dataset ud jayega.
`Procfile` gunicorn ko **1 worker** par chalata hai (zaroori hai).

## Files
```
app.py            routes (admin, employee, face verify)       database.py   SQLite + attendance/hours logic
face_utils.py     detection, training, 1:1 verification, zip  templates/     public pages, admin/, employee/
static/           css (Inter font, blue/slate theme) + js/face.js (camera)                   dataset/      face samples (dataset/<id>/n.jpg)
dataset_archive/  dataset.zip (auto)                          trainer/      trainer.yml (LBPH model)
```

## Zaroori limitations (imandari se)
* Recognition **OpenCV LBPH** hai — chhoti company ke liye theek, lekin deep-learning jitni mazboot nahi. Roshni acchi rakhein.
* **Liveness detection nahi hai:** koi employee ki *photo/video* camera ke saamne rakh de to fool ho sakta hai.
  Is liye admin approval aur face + password dono rakhe gaye hain. Sakht security chahiye to ArcFace/MediaPipe + blink check add karein.
* Ek din mein **ek check-in aur ek check-out**; raat 12 baje ke baad wali shift support nahi.
* Agar koi check-out bhool jaye to wo din admin ko **"No Check-out"** dikhta hai.
# dl


## Attendance workflow additions
- Main admin can manage office locations at `/admin/locations`, set office latitude/longitude and a permitted radius, and assign an employee to one office. The browser must share location, and check-in is blocked outside that office's configured radius. Configure accurate coordinates before using attendance restrictions in production.
- Employees use **Work from Home** to submit a request; an admin can approve/reject it. Approved WFH check-in records the browser-provided coordinates (requires location permission).
- Employees can message HR, Admin or CEO from their portal; admins can review and reply under Employee Messages.
- Attendance exports include work mode and location. Use **Export Excel** for `.xlsx` or **Export CSV** for CSV. The workbook is generated from current attendance records by the application; no Excel upload is required.
- Dataset ZIP download is restricted to the main admin. Do not share the dataset because it contains biometric face samples.
- Passwords are deliberately never exported to Excel: they are stored as password hashes, not readable passwords.


## Employee profiles and work tracker
- Employees can update their contact email, phone, designation, profile photo and professional/social links from **Profile**.
- Approved employee profiles are visible to signed-in employees and admins through **Team Directory / People Directory**. Only approved employees are listed; passwords are never public or exported.
- Employees can submit their own work proposals at `/tasks`. These remain **Proposed** until an admin reviews them.
- Admins assign work, set target dates, approve/reject proposals and track progress at `/admin/tasks`.
- Employees can update task status (Assigned / In Progress / Completed) and leave progress notes.
- Admins can download a separate `employee_work_tracker.xlsx` report from the Work & Tasks page. It includes employee, department, task, assigned-by, due date, status and latest employee update.
- New profile columns and the task table are added through safe, additive SQLite migrations at startup. Existing employee/attendance records are preserved.
- Profile image uploads accept JPG, PNG or WEBP and are limited by the application's 4 MB request limit. Install requirements after updating so Pillow is available for image validation.


## OpenCV startup troubleshooting (Windows)
If startup reports `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`, remove conflicting OpenCV packages and reinstall the package declared for this project in the same Python environment:
```powershell
python -m pip uninstall opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless -y
python -m pip install --no-cache-dir -r requirements.txt
python -c "import cv2; print(cv2.__file__); print(cv2.__version__); print(hasattr(cv2, 'CascadeClassifier'))"
```
The final line should print `True`. Also make sure the project folder does not contain a file or folder named `cv2.py` / `cv2` that shadows OpenCV.

Profile photos are stored under `DATA_DIR/profile_uploads` and served through an authenticated route, so approved employees and admins can view them while unauthenticated visitors cannot. Set `DATA_DIR` to a persistent writable volume in production.
#   f i n a l d l  
 #   t e s t i n g  
 