# ☀️ PM Surya Ghar Yojana - Solar Subsidy Platform

![React](https://img.shields.io/badge/React-18.x-blue?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)
![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue?style=for-the-badge&logo=typescript)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.x-38B2AC?style=for-the-badge&logo=tailwind-css)

An advanced, AI-powered web platform designed for the **PM Surya Ghar: Muft Bijli Yojana**. This application streamlines the process of solar panel verification, roof potential analysis, and subsidy calculation for users aiming to adopt solar energy.

---

## ✨ Key Features

- **🔐 Secure Authentication:** Seamless login via traditional Email/Password (JWT) or **Google Identity Services (OAuth)**.
- **🗺️ Interactive 3D Maps:** Integrated with **Google Maps 3D** and **ArcGIS JS API** for precise geographic visualization.
- **🤖 AI-Powered Analysis:** Utilizes computer vision (via Roboflow) to automatically detect roof types and estimate solar potential.
- **💰 Subsidy Calculator:** Dynamically calculates potential government subsidies based on user input and location.
- **🎨 Premium UI/UX:** Built with a modern "Cyber-Solar" aesthetic featuring glassmorphism, neon glows, and fluid micro-animations powered by **Framer Motion**.

---

## 🛠️ Technology Stack

### Frontend (Client)
- **Framework:** React 18 with Vite
- **Language:** TypeScript
- **Styling:** Tailwind CSS + Custom CSS (Glassmorphism)
- **Animations:** Framer Motion
- **Maps:** Google Maps JavaScript API, ArcGIS JS API 4.30
- **Auth:** Google Identity Services (GSI)

### Backend (Server)
- **Framework:** FastAPI (Python)
- **Database:** SQLite (Development) / PostgreSQL (Production) via SQLAlchemy ORM
- **Authentication:** JWT (JSON Web Tokens) & Google OAuth token verification
- **Machine Learning:** Custom model integration for roof analysis

---

## 🚀 Getting Started

Follow these steps to run the project locally on your machine.

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/2204caleb2007-mech/PMSuryaGharYojana.git
cd PMSuryaGharYojana
```

### 2. Environment Setup (.env)
Create a `.env` file in the root directory based on the following template. **Do not commit your `.env` file!**

```env
# Database Configuration
DATABASE_URL=sqlite:///./solar_analyzer.db

# JWT Security
SECRET_KEY=generate_a_random_secure_32_character_string_here

# Google Auth
GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret
VITE_GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com

# API Keys
VITE_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```

### 3. Backend Setup
Open a terminal and set up the Python environment:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server (Runs on port 8000)
python main.py
```
*Note: The SQLite database (`solar_analyzer.db`) will be created automatically on the first run.*

### 4. Frontend Setup
Open a **second terminal** and start the Vite development server:

```bash
# Install NPM packages
npm install

# Start the React app (Runs on port 5173)
npm run dev
```

### 5. View the App
Navigate to `http://localhost:5173` in your browser. The frontend is configured to automatically proxy API requests to the Python backend running on port 8000.

---

## 🔒 Security Notes
- API Keys and Client Secrets are managed strictly via environment variables.
- The repository is configured to ignore `.env` files and local databases to prevent credential leakage.

## 📄 License
This project is proprietary and intended for the specific use case of the PM Surya Ghar Yojana initiative.
