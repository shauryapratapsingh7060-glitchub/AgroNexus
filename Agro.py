import os
import math
from datetime import datetime, timedelta

import pandas as pd
from flask import (
    Flask, render_template_string, redirect, url_for, request,
    flash, jsonify, send_from_directory
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv

# Razorpay is optional at import time so the app can still open before setup.
try:
    import razorpay
except ImportError:
    razorpay = None

load_dotenv()

# ============================================================
# APP CONFIG
# ============================================================
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///agrilink.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
razorpay_client = None

if razorpay and RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
    razorpay_client = razorpay.Client(
        auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)
    )

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# ============================================================
# DATABASE MODELS
# ============================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # farmer/buyer/fpo/logistics
    location = db.Column(db.String(200), default="")
    phone = db.Column(db.String(30), default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    products = db.relationship(
        "Product", backref="farmer", lazy=True,
        foreign_keys="Product.farmer_id"
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50), default="Other")
    quantity = db.Column(db.Float, nullable=False)
    price_per_kg = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(200), default="")
    image_file = db.Column(db.String(200))
    description = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    orders = db.relationship("Order", backref="product", lazy=True)


class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(
        db.Integer, db.ForeignKey("product.id"), nullable=False
    )
    buyer_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False
    )
    quantity = db.Column(db.Float, nullable=False)
    total_price = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(30), default="Payment Pending")
    payment_status = db.Column(db.String(30), default="Pending")
    payment_id = db.Column(db.String(150))
    razorpay_order_id = db.Column(db.String(150))
    order_date = db.Column(db.DateTime, default=datetime.utcnow)
    logistics_partner_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=True
    )
    delivery_address = db.Column(db.String(300), default="")

    buyer = db.relationship(
        "User", foreign_keys=[buyer_id], backref="buyer_orders"
    )
    logistics_partner = db.relationship(
        "User", foreign_keys=[logistics_partner_id],
        backref="logistics_orders"
    )


class FPO(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    location = db.Column(db.String(200), default="")
    description = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class FPOMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fpo_id = db.Column(db.Integer, db.ForeignKey("fpo.id"), nullable=False)
    farmer_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)

    fpo = db.relationship("FPO", backref="members")
    farmer = db.relationship("User", backref="fpo_memberships")


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    message = db.Column(db.String(300), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="notifications")


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ============================================================
# DATABASE INIT
# ============================================================
with app.app_context():
    db.create_all()


# ============================================================
# HELPERS
# ============================================================
def add_notification(user_id, message):
    if user_id:
        db.session.add(
            Notification(user_id=user_id, message=message)
        )


def allowed_image(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def safe_float(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_status_steps(status):
    steps = [
        "Order Placed",
        "Paid",
        "Accepted",
        "In Transit",
        "Delivered"
    ]
    if status == "Payment Pending":
        return {"steps": steps, "current": 0}

    mapping = {
        "Pending": 1,
        "Paid": 1,
        "Accepted": 2,
        "In Transit": 3,
        "Delivered": 4
    }
    return {"steps": steps, "current": mapping.get(status, 0)}


def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


# ============================================================
# COMMON HTML
# ============================================================
BASE_HEAD = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ title or "AgriLink" }}</title>

    <link
      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
      rel="stylesheet">

    <link
      rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        body {
            background: #f6f8f7;
        }
        .hero {
            background: linear-gradient(135deg, #087f23, #36a852);
            color: white;
            border-radius: 20px;
            padding: 55px 30px;
        }
        .card {
            border: 0;
            border-radius: 16px;
            box-shadow: 0 4px 18px rgba(0,0,0,.07);
        }
        .stat-card {
            min-height: 120px;
        }
        .product-img {
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 14px 14px 0 0;
        }
        #map {
            height: 480px;
            border-radius: 16px;
        }
        .tracking-line {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .tracking-step {
            padding: 10px 14px;
            border-radius: 30px;
            background: #e9ecef;
        }
        .tracking-step.active {
            background: #198754;
            color: white;
        }
    </style>
</head>
<body>
"""

NAVBAR = """
<nav class="navbar navbar-expand-lg navbar-dark bg-success">
  <div class="container">
    <a class="navbar-brand fw-bold" href="{{ url_for('index') }}">
        🌾 AgriLink
    </a>

    <button class="navbar-toggler" type="button"
            data-bs-toggle="collapse" data-bs-target="#nav">
        <span class="navbar-toggler-icon"></span>
    </button>

    <div class="collapse navbar-collapse" id="nav">
      <div class="navbar-nav ms-auto">

        <a class="nav-link" href="{{ url_for('marketplace') }}">
            Marketplace
        </a>

        {% if current_user.is_authenticated %}
            {% if current_user.role == "farmer" %}
                <a class="nav-link"
                   href="{{ url_for('farmer_dashboard') }}">
                    Farmer Dashboard
                </a>
            {% elif current_user.role == "buyer" %}
                <a class="nav-link"
                   href="{{ url_for('buyer_dashboard') }}">
                    Buyer Dashboard
                </a>
            {% elif current_user.role == "fpo" %}
                <a class="nav-link"
                   href="{{ url_for('fpo_dashboard') }}">
                    FPO Dashboard
                </a>
            {% elif current_user.role == "logistics" %}
                <a class="nav-link"
                   href="{{ url_for('logistics_dashboard') }}">
                    Logistics
                </a>
            {% endif %}

            <a class="nav-link" href="{{ url_for('forecast') }}">
                AI Forecast
            </a>
            <a class="nav-link" href="{{ url_for('route_optimization') }}">
                Route
            </a>
            <a class="nav-link" href="{{ url_for('notifications') }}">
                🔔
            </a>
            <a class="nav-link" href="{{ url_for('logout') }}">
                Logout
            </a>
        {% else %}
            <a class="nav-link" href="{{ url_for('login') }}">
                Login
            </a>
            <a class="nav-link" href="{{ url_for('register') }}">
                Register
            </a>
        {% endif %}
      </div>
    </div>
  </div>
</nav>
"""

BASE_FOOT = """
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


def render_page(content, title="AgroNexus", **context):
    flash_html = """
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% for category, message in messages %}
        <div class="alert alert-{{ category }} alert-dismissible fade show">
            {{ message }}
            <button type="button" class="btn-close"
                    data-bs-dismiss="alert"></button>
        </div>
      {% endfor %}
    {% endwith %}
    """

    full = (
        BASE_HEAD +
        NAVBAR +
        '<main class="container py-4">' +
        flash_html +
        content +
        "</main>" +
        BASE_FOOT
    )

    return render_template_string(full, title=title, **context)


# ============================================================
# HOME
# ============================================================
@app.route("/")
def index():
    content = """
    <div class="hero text-center">
        <h1 class="display-4 fw-bold">🌾 AgriLink</h1>
        <p class="lead">
            Direct digital marketplace connecting Farmers,
            Buyers, FPOs and Logistics.
        </p>
        <a href="{{ url_for('marketplace') }}"
           class="btn btn-light btn-lg me-2">
            Browse Marketplace
        </a>
        {% if not current_user.is_authenticated %}
        <a href="{{ url_for('register') }}"
           class="btn btn-outline-light btn-lg">
            Join AgriLink
        </a>
        {% endif %}
    </div>

    <div class="row g-4 mt-4">
      <div class="col-md-3">
        <div class="card p-4 text-center h-100">
          <h2>👨‍🌾</h2><h5>Farmers</h5>
          <p>Sell crops directly and get AI-based price insights.</p>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card p-4 text-center h-100">
          <h2>🛒</h2><h5>Buyers</h5>
          <p>Find fresh products and purchase directly from farmers.</p>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card p-4 text-center h-100">
          <h2>🏢</h2><h5>FPOs</h5>
          <p>Aggregate farmer produce for bulk selling.</p>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card p-4 text-center h-100">
          <h2>🚚</h2><h5>Logistics</h5>
          <p>Track deliveries and optimize collection routes.</p>
        </div>
      </div>
    </div>
    """
    return render_page(content, "AgroNexus | Home")


# ============================================================
# AUTH
# ============================================================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        role = request.form.get("role", "")
        location = request.form.get("location", "").strip()
        phone = request.form.get("phone", "").strip()

        if role not in {"farmer", "buyer", "fpo", "logistics"}:
            flash("Invalid role.", "danger")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")
            return redirect(url_for("register"))

        user = User(
            username=username,
            email=email,
            role=role,
            location=location,
            phone=phone
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    content = """
    <div class="row justify-content-center">
      <div class="col-md-7">
        <div class="card p-4">
          <h2 class="mb-4">Create AgriLink Account</h2>

          <form method="POST">
            <div class="mb-3">
              <label class="form-label">Username</label>
              <input name="username" class="form-control" required>
            </div>

            <div class="mb-3">
              <label class="form-label">Email</label>
              <input type="email" name="email"
                     class="form-control" required>
            </div>

            <div class="mb-3">
              <label class="form-label">Phone</label>
              <input name="phone" class="form-control">
            </div>

            <div class="mb-3">
              <label class="form-label">Password</label>
              <input type="password" name="password"
                     class="form-control" minlength="6" required>
            </div>

            <div class="mb-3">
              <label class="form-label">Role</label>
              <select name="role" class="form-select" required>
                <option value="farmer">👨‍🌾 Farmer</option>
                <option value="buyer">🛒 Buyer</option>
                <option value="fpo">🏢 FPO</option>
                <option value="logistics">🚚 Logistics Partner</option>
              </select>
            </div>

            <div class="mb-3">
              <label class="form-label">Location</label>
              <input name="location"
                     placeholder="City / Village"
                     class="form-control">
            </div>

            <button class="btn btn-success w-100">
                Register
            </button>
          </form>
        </div>
      </div>
    </div>
    """
    return render_page(content, "Register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)

            if user.role == "farmer":
                return redirect(url_for("farmer_dashboard"))
            if user.role == "buyer":
                return redirect(url_for("buyer_dashboard"))
            if user.role == "fpo":
                return redirect(url_for("fpo_dashboard"))
            return redirect(url_for("logistics_dashboard"))

        flash("Invalid email or password.", "danger")

    content = """
    <div class="row justify-content-center">
      <div class="col-md-5">
        <div class="card p-4">
          <h2 class="mb-4">Login</h2>

          <form method="POST">
            <div class="mb-3">
              <label>Email</label>
              <input type="email" name="email"
                     class="form-control" required>
            </div>

            <div class="mb-3">
              <label>Password</label>
              <input type="password" name="password"
                     class="form-control" required>
            </div>

            <button class="btn btn-success w-100">
                Login
            </button>
          </form>
        </div>
      </div>
    </div>
    """
    return render_page(content, "Login")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))


# ============================================================
# FARMER
# ============================================================
@app.route("/farmer/dashboard")
@login_required
def farmer_dashboard():
    if current_user.role != "farmer":
        flash("Access denied.", "danger")
        return redirect(url_for("marketplace"))

    products = Product.query.filter_by(
        farmer_id=current_user.id
    ).order_by(Product.created_at.desc()).all()

    orders = (
        Order.query.join(Product)
        .filter(Product.farmer_id == current_user.id)
        .order_by(Order.order_date.desc())
        .all()
    )

    revenue = sum(
        o.total_price for o in orders
        if o.payment_status == "Paid"
    )

    content = """
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div>
        <h2>👨‍🌾 Farmer Dashboard</h2>
        <p class="text-muted">
          Welcome, {{ current_user.username }}
        </p>
      </div>
      <a href="{{ url_for('add_product') }}"
         class="btn btn-success">
         + Add Product
      </a>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-3">
        <div class="card stat-card p-4">
          <small>Products</small>
          <h2>{{ products|length }}</h2>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card stat-card p-4">
          <small>Orders</small>
          <h2>{{ orders|length }}</h2>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card stat-card p-4">
          <small>Revenue</small>
          <h2>₹{{ "%.2f"|format(revenue) }}</h2>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card stat-card p-4">
          <small>Location</small>
          <h5>{{ current_user.location or "Not set" }}</h5>
        </div>
      </div>
    </div>

    <div class="card p-4 mb-4">
      <h4>My Products</h4>
      <div class="table-responsive">
        <table class="table align-middle">
          <thead>
            <tr>
              <th>Name</th><th>Category</th>
              <th>Quantity</th><th>Price/kg</th>
              <th>AI Price</th>
            </tr>
          </thead>
          <tbody>
          {% for p in products %}
            <tr>
              <td>{{ p.name }}</td>
              <td>{{ p.category }}</td>
              <td>{{ "%.2f"|format(p.quantity) }} kg</td>
              <td>₹{{ "%.2f"|format(p.price_per_kg) }}</td>
              <td>
                <a class="btn btn-sm btn-outline-success"
                   href="{{ url_for('price_recommendation',
                                    product_id=p.id) }}">
                   🤖 Recommend
                </a>
              </td>
            </tr>
          {% else %}
            <tr><td colspan="5">No products yet.</td></tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <div class="card p-4">
      <h4>Orders Received</h4>
      <div class="table-responsive">
      <table class="table">
        <thead>
          <tr>
            <th>ID</th><th>Product</th><th>Buyer</th>
            <th>Qty</th><th>Payment</th><th>Status</th>
            <th>Update</th>
          </tr>
        </thead>
        <tbody>
        {% for o in orders %}
          <tr>
            <td>#{{ o.id }}</td>
            <td>{{ o.product.name }}</td>
            <td>{{ o.buyer.username }}</td>
            <td>{{ o.quantity }} kg</td>
            <td>{{ o.payment_status }}</td>
            <td>{{ o.status }}</td>
            <td>
              <form method="POST"
                    action="{{ url_for('update_order_status',
                                       order_id=o.id) }}">
                <select name="status"
                        class="form-select form-select-sm"
                        onchange="this.form.submit()">
                  <option value="Paid"
                    {% if o.status == "Paid" %}selected{% endif %}>
                    Paid
                  </option>
                  <option value="Accepted"
                    {% if o.status == "Accepted" %}selected{% endif %}>
                    Accepted
                  </option>
                  <option value="In Transit"
                    {% if o.status == "In Transit" %}selected{% endif %}>
                    In Transit
                  </option>
                  <option value="Delivered"
                    {% if o.status == "Delivered" %}selected{% endif %}>
                    Delivered
                  </option>
                </select>
              </form>
            </td>
          </tr>
        {% else %}
          <tr><td colspan="7">No orders received.</td></tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
    </div>
    """

    return render_page(content, "Farmer Dashboard",
                       products=products, orders=orders)


@app.route("/farmer/add_product", methods=["GET", "POST"])
@login_required
def add_product():
    if current_user.role != "farmer":
        flash("Access denied.", "danger")
        return redirect(url_for("marketplace"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        category = request.form.get("category", "Other")
        quantity = safe_float(request.form.get("quantity"))
        price = safe_float(request.form.get("price"))
        location = request.form.get(
            "location", current_user.location
        ).strip()
        description = request.form.get("description", "").strip()

        if not name or quantity <= 0 or price <= 0:
            flash("Enter valid product, quantity and price.", "danger")
            return redirect(url_for("add_product"))

        image_file = None
        uploaded = request.files.get("image")

        if uploaded and uploaded.filename:
            if not allowed_image(uploaded.filename):
                flash("Only PNG, JPG, JPEG and WEBP images are allowed.",
                      "danger")
                return redirect(url_for("add_product"))

            original = secure_filename(uploaded.filename)
            extension = original.rsplit(".", 1)[1].lower()
            image_file = f"{current_user.id}_{int(datetime.now().timestamp())}.{extension}"
            uploaded.save(
                os.path.join(app.config["UPLOAD_FOLDER"], image_file)
            )

        product = Product(
            farmer_id=current_user.id,
            name=name,
            category=category,
            quantity=quantity,
            price_per_kg=price,
            location=location,
            image_file=image_file,
            description=description
        )

        db.session.add(product)
        db.session.commit()

        flash("Product added successfully.", "success")
        return redirect(url_for("farmer_dashboard"))

    content = """
    <div class="row justify-content-center">
      <div class="col-md-8">
        <div class="card p-4">
          <h2>Add New Product</h2>

          <form method="POST" enctype="multipart/form-data">
            <div class="mb-3">
              <label>Product Name</label>
              <input name="name" class="form-control" required>
            </div>

            <div class="mb-3">
              <label>Category</label>
              <select name="category" class="form-select">
                <option>Vegetable</option>
                <option>Fruit</option>
                <option>Grain</option>
                <option>Pulses</option>
                <option>Other</option>
              </select>
            </div>

            <div class="row">
              <div class="col-md-6 mb-3">
                <label>Quantity (kg)</label>
                <input type="number" step="0.01" min="0.01"
                       name="quantity" class="form-control" required>
              </div>
              <div class="col-md-6 mb-3">
                <label>Price per kg (₹)</label>
                <input type="number" step="0.01" min="0.01"
                       name="price" class="form-control" required>
              </div>
            </div>

            <div class="mb-3">
              <label>Location</label>
              <input name="location"
                     value="{{ current_user.location }}"
                     class="form-control">
            </div>

            <div class="mb-3">
              <label>Image</label>
              <input type="file" name="image"
                     accept=".png,.jpg,.jpeg,.webp"
                     class="form-control">
            </div>

            <div class="mb-3">
              <label>Description</label>
              <textarea name="description"
                        class="form-control" rows="4"></textarea>
            </div>

            <button class="btn btn-success">
              Add Product
            </button>
          </form>
        </div>
      </div>
    </div>
    """
    return render_page(content, "Add Product")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"], filename
    )


# ============================================================
# MARKETPLACE
# ============================================================
@app.route("/marketplace")
def marketplace():
    search = request.args.get("search", "").strip()
    category = request.args.get("category", "").strip()

    query = Product.query.filter(Product.quantity > 0)

    if search:
        query = query.filter(Product.name.ilike(f"%{search}%"))

    if category:
        query = query.filter(Product.category == category)

    products = query.order_by(Product.created_at.desc()).all()

    content = """
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h2>🛒 Marketplace</h2>
      <span class="badge bg-success">
        {{ products|length }} products
      </span>
    </div>

    <form class="card p-3 mb-4" method="GET">
      <div class="row g-2">
        <div class="col-md-7">
          <input name="search"
                 value="{{ search }}"
                 class="form-control"
                 placeholder="Search crops, vegetables, fruits...">
        </div>
        <div class="col-md-3">
          <select name="category" class="form-select">
            <option value="">All Categories</option>
            <option {% if category=="Vegetable" %}selected{% endif %}>
              Vegetable
            </option>
            <option {% if category=="Fruit" %}selected{% endif %}>
              Fruit
            </option>
            <option {% if category=="Grain" %}selected{% endif %}>
              Grain
            </option>
            <option {% if category=="Pulses" %}selected{% endif %}>
              Pulses
            </option>
          </select>
        </div>
        <div class="col-md-2">
          <button class="btn btn-success w-100">Search</button>
        </div>
      </div>
    </form>

    <div class="row g-4">
    {% for p in products %}
      <div class="col-md-4">
        <div class="card h-100">
          {% if p.image_file %}
            <img class="product-img"
                 src="{{ url_for('uploaded_file',
                                 filename=p.image_file) }}">
          {% endif %}

          <div class="p-4">
            <span class="badge bg-success mb-2">
              {{ p.category }}
            </span>
            <h4>{{ p.name }}</h4>

            <p class="mb-1">
              <strong class="text-success">
                ₹{{ "%.2f"|format(p.price_per_kg) }}/kg
              </strong>
            </p>
            <p class="mb-1">
              Available: {{ "%.2f"|format(p.quantity) }} kg
            </p>
            <p class="mb-1">
              Farmer: {{ p.farmer.username }}
            </p>
            <p class="text-muted">
              📍 {{ p.location or "N/A" }}
            </p>

            {% if p.description %}
              <p>{{ p.description }}</p>
            {% endif %}

            {% if current_user.is_authenticated
                  and current_user.role == "buyer" %}
              <a href="{{ url_for('order', product_id=p.id) }}"
                 class="btn btn-success w-100">
                 Order Now
              </a>
            {% elif not current_user.is_authenticated %}
              <a href="{{ url_for('login') }}"
                 class="btn btn-outline-success w-100">
                 Login to Order
              </a>
            {% endif %}
          </div>
        </div>
      </div>
    {% else %}
      <div class="col-12">
        <div class="alert alert-info">
          No products found.
        </div>
      </div>
    {% endfor %}
    </div>
    """

    return render_page(
        content,
        "Marketplace",
        products=products,
        search=search,
        category=category
    )


# ============================================================
# ORDER CREATION
# ============================================================
@app.route("/order/<int:product_id>", methods=["GET", "POST"])
@login_required
def order(product_id):
    if current_user.role != "buyer":
        flash("Only buyers can place orders.", "danger")
        return redirect(url_for("marketplace"))

    product = db.session.get(Product, product_id)

    if not product:
        flash("Product not found.", "danger")
        return redirect(url_for("marketplace"))

    if request.method == "POST":
        quantity = safe_float(request.form.get("quantity"))
        address = request.form.get("address", "").strip()

        if quantity <= 0:
            flash("Quantity must be greater than zero.", "danger")
            return redirect(url_for("order", product_id=product.id))

        if quantity > product.quantity:
            flash("Not enough stock available.", "danger")
            return redirect(url_for("order", product_id=product.id))

        if not address:
            flash("Delivery address is required.", "danger")
            return redirect(url_for("order", product_id=product.id))

        total = round(quantity * product.price_per_kg, 2)

        new_order = Order(
            product_id=product.id,
            buyer_id=current_user.id,
            quantity=quantity,
            total_price=total,
            status="Payment Pending",
            payment_status="Pending",
            delivery_address=address
        )

        # Reserve stock at order creation.
        product.quantity -= quantity

        db.session.add(new_order)
        db.session.flush()

        add_notification(
            product.farmer_id,
            f"New order #{new_order.id} received for {product.name}."
        )

        db.session.commit()

        return redirect(url_for(
            "payment", order_id=new_order.id
        ))

    content = """
    <div class="row justify-content-center">
      <div class="col-md-7">
        <div class="card p-4">
          <h2>Place Order</h2>

          <h4>{{ product.name }}</h4>
          <p>Price: ₹{{ "%.2f"|format(product.price_per_kg) }}/kg</p>
          <p>Available: {{ "%.2f"|format(product.quantity) }} kg</p>

          <form method="POST">
            <div class="mb-3">
              <label>Quantity (kg)</label>
              <input type="number" name="quantity"
                     step="0.01" min="0.01"
                     max="{{ product.quantity }}"
                     class="form-control" required>
            </div>

            <div class="mb-3">
              <label>Delivery Address</label>
              <textarea name="address"
                        class="form-control"
                        required></textarea>
            </div>

            <button class="btn btn-success w-100">
              Continue to Payment
            </button>
          </form>
        </div>
      </div>
    </div>
    """
    return render_page(content, "Place Order", product=product)


# ============================================================
# RAZORPAY
# ============================================================
@app.route("/payment/<int:order_id>")
@login_required
def payment(order_id):
    order = db.session.get(Order, order_id)

    if not order or order.buyer_id != current_user.id:
        flash("Access denied.", "danger")
        return redirect(url_for("buyer_dashboard"))

    if order.payment_status == "Paid":
        return redirect(url_for("track_order", order_id=order.id))

    if not razorpay_client:
        flash(
            "Razorpay is not configured. Add RAZORPAY_KEY_ID and "
            "RAZORPAY_KEY_SECRET to .env.",
            "warning"
        )
        return redirect(url_for("buyer_dashboard"))

    try:
        razor_order = razorpay_client.order.create({
            "amount": int(round(order.total_price * 100)),
            "currency": "INR",
            "receipt": f"AGRI-{order.id}",
            "notes": {
                "order_id": str(order.id),
                "product": order.product.name
            }
        })

        order.razorpay_order_id = razor_order["id"]
        db.session.commit()

    except Exception as exc:
        flash(f"Unable to create Razorpay order: {exc}", "danger")
        return redirect(url_for("buyer_dashboard"))

    content = """
    <div class="row justify-content-center">
      <div class="col-md-6">
        <div class="card p-4 text-center">
          <h2>💳 AgriLink Payment</h2>
          <hr>
          <h5>{{ order.product.name }}</h5>
          <p>Quantity: {{ order.quantity }} kg</p>
          <h2 class="text-success">
            ₹{{ "%.2f"|format(order.total_price) }}
          </h2>

          <button id="pay-btn" class="btn btn-success btn-lg">
            Pay with Razorpay
          </button>

          <p class="small text-muted mt-3">
            Prototype uses Razorpay. Use Test Mode for demo.
          </p>
        </div>
      </div>
    </div>

    <script src="https://checkout.razorpay.com/v1/checkout.js"></script>
    <script>
      const options = {
        key: {{ key_id|tojson }},
        amount: {{ razor_order.amount|tojson }},
        currency: "INR",
        name: "AgriLink",
        description: "Agricultural Product Order",
        order_id: {{ razor_order.id|tojson }},

        prefill: {
          name: {{ current_user.username|tojson }},
          email: {{ current_user.email|tojson }},
          contact: {{ current_user.phone|tojson }}
        },

        notes: {
          agri_order_id: {{ order.id|tojson }}
        },

        theme: {
          color: "#198754"
        },

        handler: function (response) {
          fetch("{{ url_for('payment_success') }}", {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify(response)
          })
          .then(async response => {
            const data = await response.json();

            if (data.success) {
              window.location.href =
                "{{ url_for('track_order',
                            order_id=order.id) }}";
            } else {
              alert(data.message || "Payment verification failed.");
            }
          })
          .catch(() => {
            alert("Payment verification request failed.");
          });
        }
      };

      const rzp = new Razorpay(options);

      document.getElementById("pay-btn").onclick = function(e) {
        rzp.open();
        e.preventDefault();
      };
    </script>
    """# Razorpay order create karein (amount paise me hota hai, isliye * 100)
    amount_in_paise = int(order.total_amount * 100)
    
    raz_order = razorpay_client.order.create({
        "amount": amount_in_paise,
        "currency": "INR",
        "payment_capture": "1"
    })

    return render_page(
        content,
        "Razorpay Payment",
        order=order,
        razor_order=raz_order,
        key_id=RAZORPAY_KEY_ID
    )


@app.route("/payment-success", methods=["POST"])
@login_required
def payment_success():
    if current_user.role != "buyer":
        return jsonify({
            "success": False,
            "message": "Only buyers can verify payment."
        }), 403

    data = request.get_json(silent=True) or {}

    required = [
        "razorpay_order_id",
        "razorpay_payment_id",
        "razorpay_signature"
    ]

    if not all(data.get(k) for k in required):
        return jsonify({
            "success": False,
            "message": "Incomplete payment response."
        }), 400

    if not razorpay_client:
        return jsonify({
            "success": False,
            "message": "Razorpay is not configured."
        }), 500

    order = Order.query.filter_by(
        razorpay_order_id=data["razorpay_order_id"],
        buyer_id=current_user.id
    ).first()

    if not order:
        return jsonify({
            "success": False,
            "message": "Order not found."
        }), 404

    try:
        razorpay_client.utility.verify_payment_signature({
            "razorpay_order_id": data["razorpay_order_id"],
            "razorpay_payment_id": data["razorpay_payment_id"],
            "razorpay_signature": data["razorpay_signature"]
        })

        order.payment_id = data["razorpay_payment_id"]
        order.payment_status = "Paid"
        order.status = "Paid"

        add_notification(
            order.product.farmer_id,
            f"Payment received for order #{order.id}."
        )

        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Payment verified successfully."
        })

    except Exception:
        return jsonify({
            "success": False,
            "message": "Payment signature verification failed."
        }), 400


# ============================================================
# BUYER DASHBOARD
# ============================================================
@app.route("/buyer/dashboard")
@login_required
def buyer_dashboard():
    if current_user.role != "buyer":
        flash("Access denied.", "danger")
        return redirect(url_for("marketplace"))

    orders = (
        Order.query
        .filter_by(buyer_id=current_user.id)
        .order_by(Order.order_date.desc())
        .all()
    )

    paid_orders = [
        o for o in orders if o.payment_status == "Paid"
    ]

    spending = sum(o.total_price for o in paid_orders)
    active = [
        o for o in orders
        if o.status not in {"Delivered"}
    ]
    delivered = [
        o for o in orders if o.status == "Delivered"
    ]

    recommended = Product.query.filter(
        Product.quantity > 0
    ).order_by(Product.created_at.desc()).limit(6).all()

    content = """
    <div class="mb-4">
      <h2>🛒 Buyer Dashboard</h2>
      <p class="text-muted">
        Welcome, {{ current_user.username }}
      </p>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-3">
        <div class="card stat-card p-4">
          <small>Total Orders</small>
          <h2>{{ orders|length }}</h2>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card stat-card p-4">
          <small>Active</small>
          <h2>{{ active|length }}</h2>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card stat-card p-4">
          <small>Delivered</small>
          <h2>{{ delivered|length }}</h2>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card stat-card p-4">
          <small>Total Spending</small>
          <h2>₹{{ "%.2f"|format(spending) }}</h2>
        </div>
      </div>
    </div>

    <div class="card p-4 mb-4">
      <div class="d-flex justify-content-between">
        <h4>Recommended Products</h4>
        <a href="{{ url_for('marketplace') }}">View All</a>
      </div>

      <div class="row g-3 mt-1">
      {% for p in recommended %}
        <div class="col-md-4">
          <div class="border rounded p-3 h-100">
            <span class="badge bg-success">{{ p.category }}</span>
            <h5 class="mt-2">{{ p.name }}</h5>
            <p>₹{{ "%.2f"|format(p.price_per_kg) }}/kg</p>
            <a class="btn btn-sm btn-success"
               href="{{ url_for('order', product_id=p.id) }}">
               Buy
            </a>
          </div>
        </div>
      {% endfor %}
      </div>
    </div>

    <div class="card p-4">
      <h4>My Orders</h4>
      <div class="table-responsive">
      <table class="table align-middle">
        <thead>
          <tr>
            <th>ID</th><th>Product</th><th>Total</th>
            <th>Payment</th><th>Status</th><th>Action</th>
          </tr>
        </thead>
        <tbody>
        {% for o in orders %}
          <tr>
            <td>#{{ o.id }}</td>
            <td>{{ o.product.name }}</td>
            <td>₹{{ "%.2f"|format(o.total_price) }}</td>
            <td>{{ o.payment_status }}</td>
            <td>{{ o.status }}</td>
            <td>
              {% if o.payment_status != "Paid" %}
                <a class="btn btn-sm btn-success"
                   href="{{ url_for('payment',
                                    order_id=o.id) }}">
                   Pay
                </a>
              {% else %}
                <a class="btn btn-sm btn-outline-success"
                   href="{{ url_for('track_order',
                                    order_id=o.id) }}">
                   Track
                </a>
              {% endif %}
            </td>
          </tr>
        {% else %}
          <tr><td colspan="6">No orders yet.</td></tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
    </div>
    """

    return render_page(
        content,
        "Buyer Dashboard",
        orders=orders,
        active=active,
        delivered=delivered,
        spending=spending,
        recommended=recommended
    )


# ============================================================
# ORDER TRACKING
# ============================================================
@app.route("/track/<int:order_id>")
@login_required
def track_order(order_id):
    order = db.session.get(Order, order_id)

    if not order:
        flash("Order not found.", "danger")
        return redirect(url_for("buyer_dashboard"))

    allowed = (
        order.buyer_id == current_user.id or
        order.product.farmer_id == current_user.id or
        order.logistics_partner_id == current_user.id
    )

    if not allowed:
        flash("Access denied.", "danger")
        return redirect(url_for("marketplace"))

    tracking = get_status_steps(order.status)

    content = """
    <div class="card p-4">
      <h2>📦 Track Order #{{ order.id }}</h2>

      <div class="row mt-4">
        <div class="col-md-6">
          <h5>{{ order.product.name }}</h5>
          <p>Quantity: {{ order.quantity }} kg</p>
          <p>Total: ₹{{ "%.2f"|format(order.total_price) }}</p>
          <p>Payment: <strong>{{ order.payment_status }}</strong></p>
          <p>Delivery: {{ order.delivery_address }}</p>
        </div>

        <div class="col-md-6">
          <h5>Order Status</h5>
          <div class="tracking-line mt-3">
          {% for step in tracking.steps %}
            <span class="tracking-step
              {% if loop.index0 <= tracking.current %}
                active
              {% endif %}">
              {% if loop.index0 <= tracking.current %}✓{% endif %}
              {{ step }}
            </span>
          {% endfor %}
          </div>
        </div>
      </div>
    </div>
    """

    return render_page(
        content,
        f"Track Order #{order.id}",
        order=order,
        tracking=tracking
    )


@app.route("/my_orders")
@login_required
def my_orders():
    if current_user.role == "buyer":
        return redirect(url_for("buyer_dashboard"))
    if current_user.role == "farmer":
        return redirect(url_for("farmer_dashboard"))
    if current_user.role == "logistics":
        return redirect(url_for("logistics_dashboard"))
    return redirect(url_for("fpo_dashboard"))


# ============================================================
# ORDER STATUS + LOGISTICS ASSIGNMENT
# ============================================================
@app.route("/update_order_status/<int:order_id>", methods=["POST"])
@login_required
def update_order_status(order_id):
    order = db.session.get(Order, order_id)

    if not order:
        flash("Order not found.", "danger")
        return redirect(url_for("marketplace"))

    if current_user.role == "farmer":
        if order.product.farmer_id != current_user.id:
            flash("Access denied.", "danger")
            return redirect(url_for("marketplace"))

    elif current_user.role == "logistics":
        if order.logistics_partner_id != current_user.id:
            flash("Access denied.", "danger")
            return redirect(url_for("logistics_dashboard"))

    else:
        flash("You cannot update this order.", "danger")
        return redirect(url_for("marketplace"))

    new_status = request.form.get("status")

    allowed_statuses = {
        "Paid", "Accepted", "In Transit", "Delivered"
    }

    if new_status not in allowed_statuses:
        flash("Invalid status.", "danger")
        return redirect(request.referrer or url_for("marketplace"))

    if new_status == "Accepted" and order.payment_status != "Paid":
        flash("Order must be paid before acceptance.", "warning")
        return redirect(request.referrer or url_for("marketplace"))

    order.status = new_status

    if new_status == "Delivered":
        add_notification(
            order.buyer_id,
            f"Order #{order.id} has been delivered."
        )

    if new_status == "In Transit":
        add_notification(
            order.buyer_id,
            f"Order #{order.id} is now in transit."
        )

    db.session.commit()

    flash("Order status updated.", "success")
    return redirect(request.referrer or url_for("marketplace"))


@app.route("/assign-logistics/<int:order_id>", methods=["POST"])
@login_required
def assign_logistics(order_id):
    if current_user.role not in {"farmer", "fpo"}:
        flash("Access denied.", "danger")
        return redirect(url_for("marketplace"))

    order = db.session.get(Order, order_id)

    if not order:
        flash("Order not found.", "danger")
        return redirect(url_for("marketplace"))

    if current_user.role == "farmer":
        if order.product.farmer_id != current_user.id:
            flash("Access denied.", "danger")
            return redirect(url_for("farmer_dashboard"))

    logistics_id = safe_int(request.form.get("logistics_id"))
    partner = User.query.filter_by(
        id=logistics_id, role="logistics"
    ).first()

    if not partner:
        flash("Invalid logistics partner.", "danger")
        return redirect(request.referrer or url_for("marketplace"))

    order.logistics_partner_id = partner.id

    add_notification(
        partner.id,
        f"Order #{order.id} assigned to you for delivery."
    )

    db.session.commit()

    flash("Logistics partner assigned.", "success")
    return redirect(request.referrer or url_for("marketplace"))


# ============================================================
# AI PRICE RECOMMENDATION
# ============================================================
@app.route("/price-recommendation/<int:product_id>")
@login_required
def price_recommendation(product_id):
    product = db.session.get(Product, product_id)

    if not product or product.farmer_id != current_user.id:
        flash("Access denied.", "danger")
        return redirect(url_for("farmer_dashboard"))

    # Estimate demand from paid orders of the same category.
    category_products = Product.query.filter_by(
        category=product.category
    ).all()

    category_ids = [p.id for p in category_products]

    paid_orders = Order.query.filter(
        Order.product_id.in_(category_ids),
        Order.payment_status == "Paid"
    ).all() if category_ids else []

    total_demand = sum(o.quantity for o in paid_orders)

    # Simple market baseline from currently listed products.
    prices = [
        p.price_per_kg for p in category_products
        if p.price_per_kg > 0
    ]

    avg_market_price = (
        sum(prices) / len(prices)
        if prices else product.price_per_kg
    )

    # Prototype AI-style recommendation:
    # demand pressure slightly adjusts the market average.
    demand_factor = 1.0

    if total_demand > 500:
        demand_factor = 1.10
    elif total_demand > 200:
        demand_factor = 1.06
    elif total_demand > 50:
        demand_factor = 1.03
    elif total_demand < 10:
        demand_factor = 0.97

    recommended = round(
        max(1, avg_market_price * demand_factor), 2
    )

    if total_demand > 200:
        demand_level = "HIGH"
    elif total_demand > 50:
        demand_level = "MEDIUM"
    else:
        demand_level = "LOW"

    estimated_revenue = round(
        recommended * product.quantity, 2
    )

    content = """
    <div class="row justify-content-center">
      <div class="col-md-8">
        <div class="card p-4">
          <h2>🤖 AI Price Recommendation</h2>
          <p class="text-muted">
            Prototype recommendation based on marketplace prices
            and historical paid demand.
          </p>

          <div class="row g-3 mt-2">
            <div class="col-md-6">
              <div class="border rounded p-3">
                <small>Product</small>
                <h4>{{ product.name }}</h4>
              </div>
            </div>

            <div class="col-md-6">
              <div class="border rounded p-3">
                <small>Current Price</small>
                <h4>₹{{ "%.2f"|format(product.price_per_kg) }}/kg</h4>
              </div>
            </div>

            <div class="col-md-6">
              <div class="border rounded p-3">
                <small>Average Market Price</small>
                <h4>₹{{ "%.2f"|format(avg_market_price) }}/kg</h4>
              </div>
            </div>

            <div class="col-md-6">
              <div class="border rounded p-3">
                <small>Demand</small>
                <h4>{{ demand_level }}</h4>
              </div>
            </div>
          </div>

          <div class="alert alert-success mt-4">
            <h3>Recommended Price:
              ₹{{ "%.2f"|format(recommended) }}/kg
            </h3>
            <p class="mb-0">
              Estimated revenue for available stock:
              ₹{{ "%.2f"|format(estimated_revenue) }}
            </p>
          </div>
        </div>
      </div>
    </div>
    """

    return render_page(
        content,
        "AI Price Recommendation",
        product=product,
        avg_market_price=avg_market_price,
        demand_level=demand_level,
        recommended=recommended,
        estimated_revenue=estimated_revenue
    )


# ============================================================
# AI DEMAND FORECAST
# ============================================================
@app.route("/forecast")
@login_required
def forecast():
    # Build daily paid order demand from real database data.
    paid_orders = Order.query.filter_by(
        payment_status="Paid"
    ).all()

    rows = []

    for order in paid_orders:
        rows.append({
            "date": order.order_date.date(),
            "demand": order.quantity
        })

    if rows:
        df = pd.DataFrame(rows)
        df = df.groupby("date", as_index=False)["demand"].sum()
    else:
        # Demo fallback if no orders exist yet.
        dates = pd.date_range(
            end=datetime.today().date(),
            periods=30
        )
        demand = [
            70 + ((i * 7) % 45)
            for i in range(len(dates))
        ]
        df = pd.DataFrame({
            "date": dates,
            "demand": demand
        })

    df["date"] = pd.to_datetime(df["date"])
    df["day_index"] = range(len(df))

    model = LinearRegression()

    if len(df) >= 2:
        model.fit(df[["day_index"]], df["demand"])

        future_indices = [
            len(df) + i for i in range(7)
        ]

        predictions = model.predict(
            pd.DataFrame({"day_index": future_indices})
        )
    else:
        predictions = [df["demand"].iloc[-1]] * 7

    predictions = [
        round(max(0, float(x)), 2)
        for x in predictions
    ]

    future_dates = [
        (datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, 8)
    ]

    content = """
    <div class="card p-4">
      <h2>🤖 AI Demand Forecast</h2>
      <p class="text-muted">
        Forecast for the next 7 days using Linear Regression.
        Real paid-order data is used when available.
      </p>

      <canvas id="forecastChart" height="100"></canvas>

      <div class="row mt-4">
      {% for i in range(predictions|length) %}
        <div class="col">
          <div class="border rounded p-2 text-center">
            <small>{{ future_dates[i] }}</small>
            <h5>{{ predictions[i] }} kg</h5>
          </div>
        </div>
      {% endfor %}
      </div>
    </div>

    <script>
      new Chart(
        document.getElementById("forecastChart"),
        {
          type: "line",
          data: {
            labels: {{ future_dates|tojson }},
            datasets: [{
              label: "Predicted Demand (kg)",
              data: {{ predictions|tojson }},
              tension: 0.2,
              borderWidth: 3
            }]
          },
          options: {
            responsive: true
          }
        }
      );
    </script>
    """

    return render_page(
        content,
        "AI Demand Forecast",
        predictions=predictions,
        future_dates=future_dates
    )


# ============================================================
# FPO MODULE
# ============================================================
@app.route("/fpo/dashboard")
@login_required
def fpo_dashboard():
    if current_user.role != "fpo":
        flash("Access denied.", "danger")
        return redirect(url_for("marketplace"))

    fpos = FPO.query.order_by(FPO.created_at.desc()).all()

    total_members = sum(len(f.members) for f in fpos)

    total_aggregated = 0

    for fpo in fpos:
        for member in fpo.members:
            farmer_products = Product.query.filter_by(
                farmer_id=member.farmer_id
            ).all()
            total_aggregated += sum(
                p.quantity for p in farmer_products
            )

    farmers = User.query.filter_by(role="farmer").all()

    content = """
    <div class="mb-4">
      <h2>🏢 FPO Dashboard</h2>
      <p class="text-muted">
        Farmer Producer Organisation and crop aggregation.
      </p>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-4">
        <div class="card p-4">
          <small>FPOs</small>
          <h2>{{ fpos|length }}</h2>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card p-4">
          <small>Farmer Members</small>
          <h2>{{ total_members }}</h2>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card p-4">
          <small>Available Farmer Stock</small>
          <h2>{{ "%.2f"|format(total_aggregated) }} kg</h2>
        </div>
      </div>
    </div>

    <div class="card p-4 mb-4">
      <h4>Create FPO</h4>
      <form method="POST"
            action="{{ url_for('create_fpo') }}">
        <div class="row">
          <div class="col-md-5 mb-2">
            <input name="name"
                   class="form-control"
                   placeholder="FPO Name" required>
          </div>
          <div class="col-md-4 mb-2">
            <input name="location"
                   class="form-control"
                   placeholder="Location">
          </div>
          <div class="col-md-3 mb-2">
            <button class="btn btn-success w-100">
              Create FPO
            </button>
          </div>
        </div>
      </form>
    </div>

    <div class="card p-4 mb-4">
      <h4>Add Farmer to FPO</h4>
      <form method="POST"
            action="{{ url_for('add_fpo_member') }}">
        <div class="row">
          <div class="col-md-6 mb-2">
            <select name="fpo_id" class="form-select" required>
              <option value="">Select FPO</option>
              {% for f in fpos %}
                <option value="{{ f.id }}">{{ f.name }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="col-md-6 mb-2">
            <select name="farmer_id" class="form-select" required>
              <option value="">Select Farmer</option>
              {% for farmer in farmers %}
                <option value="{{ farmer.id }}">
                  {{ farmer.username }} - {{ farmer.location }}
                </option>
              {% endfor %}
            </select>
          </div>
        </div>
        <button class="btn btn-outline-success mt-2">
          Add Member
        </button>
      </form>
    </div>

    <div class="row g-4">
    {% for f in fpos %}
      <div class="col-md-6">
        <div class="card p-4 h-100">
          <h4>{{ f.name }}</h4>
          <p>📍 {{ f.location or "N/A" }}</p>
          <h6>Members: {{ f.members|length }}</h6>

          <ul>
          {% for member in f.members %}
            <li>{{ member.farmer.username }}</li>
          {% else %}
            <li>No members yet.</li>
          {% endfor %}
          </ul>
        </div>
      </div>
    {% else %}
      <div class="col-12">
        <div class="alert alert-info">
          No FPO created yet.
        </div>
      </div>
    {% endfor %}
    </div>
    """

    return render_page(
        content,
        "FPO Dashboard",
        fpos=fpos,
        farmers=farmers,
        total_members=total_members,
        total_aggregated=total_aggregated
    )


@app.route("/fpo/create", methods=["POST"])
@login_required
def create_fpo():
    if current_user.role != "fpo":
        flash("Access denied.", "danger")
        return redirect(url_for("marketplace"))

    name = request.form.get("name", "").strip()
    location = request.form.get("location", "").strip()

    if not name:
        flash("FPO name is required.", "danger")
        return redirect(url_for("fpo_dashboard"))

    fpo = FPO(name=name, location=location)
    db.session.add(fpo)
    db.session.commit()

    flash("FPO created successfully.", "success")
    return redirect(url_for("fpo_dashboard"))


@app.route("/fpo/add-member", methods=["POST"])
@login_required
def add_fpo_member():
    if current_user.role != "fpo":
        flash("Access denied.", "danger")
        return redirect(url_for("marketplace"))

    fpo_id = safe_int(request.form.get("fpo_id"))
    farmer_id = safe_int(request.form.get("farmer_id"))

    fpo = db.session.get(FPO, fpo_id)
    farmer = User.query.filter_by(
        id=farmer_id, role="farmer"
    ).first()

    if not fpo or not farmer:
        flash("Invalid FPO or farmer.", "danger")
        return redirect(url_for("fpo_dashboard"))

    existing = FPOMember.query.filter_by(
        fpo_id=fpo.id,
        farmer_id=farmer.id
    ).first()

    if existing:
        flash("Farmer is already a member.", "warning")
        return redirect(url_for("fpo_dashboard"))

    db.session.add(
        FPOMember(
            fpo_id=fpo.id,
            farmer_id=farmer.id
        )
    )

    add_notification(
        farmer.id,
        f"You have been added to FPO: {fpo.name}."
    )

    db.session.commit()

    flash("Farmer added to FPO.", "success")
    return redirect(url_for("fpo_dashboard"))


# ============================================================
# LOGISTICS DASHBOARD
# ============================================================
@app.route("/logistics/dashboard")
@login_required
def logistics_dashboard():
    if current_user.role != "logistics":
        flash("Access denied.", "danger")
        return redirect(url_for("marketplace"))

    orders = Order.query.filter_by(
        logistics_partner_id=current_user.id
    ).order_by(Order.order_date.desc()).all()

    pending = [
        o for o in orders
        if o.status in {"Paid", "Accepted"}
    ]

    active = [
        o for o in orders if o.status == "In Transit"
    ]

    delivered = [
        o for o in orders if o.status == "Delivered"
    ]

    content = """
    <div class="mb-4">
      <h2>🚚 Logistics Dashboard</h2>
      <p class="text-muted">
        Manage assigned agricultural deliveries.
      </p>
    </div>

    <div class="row g-3 mb-4">
      <div class="col-md-4">
        <div class="card p-4">
          <small>Pending</small>
          <h2>{{ pending|length }}</h2>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card p-4">
          <small>In Transit</small>
          <h2>{{ active|length }}</h2>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card p-4">
          <small>Delivered</small>
          <h2>{{ delivered|length }}</h2>
        </div>
      </div>
    </div>

    <div class="card p-4">
      <h4>Assigned Deliveries</h4>

      <div class="table-responsive">
      <table class="table align-middle">
        <thead>
          <tr>
            <th>Order</th><th>Product</th>
            <th>Farmer</th><th>Buyer</th>
            <th>Address</th><th>Status</th>
            <th>Update</th>
          </tr>
        </thead>

        <tbody>
        {% for o in orders %}
          <tr>
            <td>#{{ o.id }}</td>
            <td>{{ o.product.name }}</td>
            <td>{{ o.product.farmer.username }}</td>
            <td>{{ o.buyer.username }}</td>
            <td>{{ o.delivery_address }}</td>
            <td>{{ o.status }}</td>
            <td>
              <form method="POST"
                    action="{{ url_for('update_order_status',
                                       order_id=o.id) }}">
                <select name="status"
                        class="form-select form-select-sm"
                        onchange="this.form.submit()">
                  <option value="Accepted"
                    {% if o.status=="Accepted" %}selected{% endif %}>
                    Accepted
                  </option>
                  <option value="In Transit"
                    {% if o.status=="In Transit" %}selected{% endif %}>
                    In Transit
                  </option>
                  <option value="Delivered"
                    {% if o.status=="Delivered" %}selected{% endif %}>
                    Delivered
                  </option>
                </select>
              </form>
            </td>
          </tr>
        {% else %}
          <tr>
            <td colspan="7">No assigned deliveries.</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
      </div>
    </div>
    """

    return render_page(
        content,
        "Logistics Dashboard",
        orders=orders,
        pending=pending,
        active=active,
        delivered=delivered
    )


# ============================================================
# ROUTE OPTIMIZATION + MAP
# ============================================================
@app.route("/route_optimization")
@login_required
def route_optimization():
    # Demo coordinates. In production, use geocoded user addresses.
    points = [
        {
            "name": "Farmer A",
            "lat": 28.6139,
            "lng": 77.2090,
            "type": "Farmer"
        },
        {
            "name": "Farmer B",
            "lat": 28.6129,
            "lng": 77.2295,
            "type": "Farmer"
        },
        {
            "name": "Buyer 1",
            "lat": 28.6304,
            "lng": 77.2177,
            "type": "Buyer"
        },
        {
            "name": "Buyer 2",
            "lat": 28.6218,
            "lng": 77.2350,
            "type": "Buyer"
        }
    ]

    # Nearest-neighbour route.
    remaining = points[1:].copy()
    route = [points[0]]
    current = points[0]
    total_distance = 0

    while remaining:
        nearest = min(
            remaining,
            key=lambda p: haversine(
                current["lat"], current["lng"],
                p["lat"], p["lng"]
            )
        )

        total_distance += haversine(
            current["lat"], current["lng"],
            nearest["lat"], nearest["lng"]
        )

        route.append(nearest)
        current = nearest
        remaining.remove(nearest)

    route_json = [
        [p["lat"], p["lng"], p["name"], p["type"]]
        for p in route
    ]

    content = """
    <div class="mb-3">
      <h2>🚚 Route Optimization</h2>
      <p class="text-muted">
        Demo route generated using nearest-neighbour optimization.
      </p>
    </div>

    <div class="row g-4">
      <div class="col-md-8">
        <div class="card p-2">
          <div id="map"></div>
        </div>
      </div>

      <div class="col-md-4">
        <div class="card p-4">
          <h4>Optimized Route</h4>
          <ol>
          {% for p in route %}
            <li class="mb-2">
              <strong>{{ p.name }}</strong>
              <small class="text-muted">({{ p.type }})</small>
            </li>
          {% endfor %}
          </ol>

          <hr>

          <h5>
            Approx. Distance:
            {{ "%.2f"|format(total_distance) }} km
          </h5>
        </div>
      </div>
    </div>

    <script>
      const route = {{ route_json|tojson }};

      const map = L.map("map").setView(
        [route[0][0], route[0][1]], 13
      );

      L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
          maxZoom: 19,
          attribution: "&copy; OpenStreetMap contributors"
        }
      ).addTo(map);

      const latLngs = [];

      route.forEach((point, index) => {
        const lat = point[0];
        const lng = point[1];
        const name = point[2];
        const type = point[3];

        latLngs.push([lat, lng]);

        L.marker([lat, lng])
          .addTo(map)
          .bindPopup(
            "<strong>" + (index + 1) + ". " +
            name + "</strong><br>" + type
          );
      });

      L.polyline(latLngs, {
        weight: 5
      }).addTo(map);

      map.fitBounds(latLngs);
    </script>
    """

    return render_page(
        content,
        "Route Optimization",
        route=route,
        route_json=route_json,
        total_distance=total_distance
    )

# NOTIFICATION
@app.route("/notifications")
@login_required
def notifications():
    items = Notification.query.filter_by(
        user_id=current_user.id
    ).order_by(Notification.created_at.desc()).all()

    for item in items:
        item.is_read = True

    db.session.commit()

    content = """
    <div class="card p-4">
      <h2>🔔 Notifications</h2>

      <div class="list-group mt-3">
      {% for n in items %}
        <div class="list-group-item">
          <strong>{{ n.message }}</strong>
          <br>
          <small class="text-muted">
            {{ n.created_at.strftime("%Y-%m-%d %H:%M") }}
          </small>
        </div>
      {% else %}
        <div class="alert alert-info">
          No notifications.
        </div>
      {% endfor %}
      </div>
    </div>
    """

    return render_page(
        content,
        "Notifications",
        items=items
    )
@app.route("/health")
def health():
    return jsonify({
        "app": "AgriLink",
        "status": "running",
        "razorpay_configured": bool(razorpay_client)
    })


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000))
    )
