# How to Run the Solar Panel Analyzer

## Quick Start (Step by Step)

### Step 1: Install Dependencies

1. **Activate your virtual environment** (if you have one):
   ```powershell
   venv\Scripts\activate
   ```

2. **Install all required packages**:
   ```powershell
   pip install -r requirements.txt
   ```

### Step 2: Set Up PostgreSQL Database

1. **Install PostgreSQL** (if not already installed):
   - Download from: https://www.postgresql.org/download/windows/
   - Install with default settings

2. **Create the database**:
   - Open pgAdmin or psql
   - Run: `CREATE DATABASE solar_analyzer;`

3. **Note your database credentials**:
   - Default username: `postgres`
   - Default password: (the one you set during installation)
   - Default port: `5432`

### Step 3: Configure Environment Variables

1. **Create a `.env` file** in the project root (same directory as `main.py`)

2. **Add the following content** (replace with your values):

```env
# Database - UPDATE WITH YOUR POSTGRESQL CREDENTIALS
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/solar_analyzer

# JWT Secret Key - CHANGE THIS TO A RANDOM STRING (min 32 characters)
SECRET_KEY=your-super-secret-key-change-this-in-production-min-32-characters-long

# Application Settings
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Optional: External APIs (leave empty if not using)
GOOGLE_MAPS_API_KEY=
NASA_POWER_API_KEY=
ESRI_API_KEY=

# ML Models
ML_MODELS_DIR=model_weights
ML_MODEL_PATH=model_weights/solar_detector.pth
```

**Important**: Replace `YOUR_PASSWORD` with your PostgreSQL password, and change `SECRET_KEY` to a random string.

### Step 4: Run the Application

**Option 1: Using Python directly**
```powershell
python main.py
```

**Option 2: Using uvicorn directly**
```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

You should see output like:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 5: Access the Application

1. **Open your web browser**
2. **Navigate to**: `http://localhost:8000`
3. **You should see the landing page in guest mode**

### Step 6: Test the Application

1. **Register a new account**:
   - Click "Register" (or navigate to registration)
   - Fill in email, username, password
   - Submit the form

2. **Login**:
   - Use your credentials to log in
   - You should receive a JWT token (stored in browser)

3. **Run an analysis**:
   - Go to "Solar Analysis" page
   - Enter latitude and longitude
   - Click "Analyze Location"
   - The ML model will analyze satellite imagery

## Troubleshooting

### Error: "Module not found" or Import errors
**Solution**: Make sure virtual environment is activated and dependencies are installed:
```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

### Error: "Could not connect to database" or PostgreSQL errors
**Solutions**:
- Ensure PostgreSQL service is running (check Windows Services)
- Verify DATABASE_URL in `.env` is correct
- Check that database `solar_analyzer` exists
- Verify username/password are correct

### Error: "Port 8000 already in use"
**Solution**: Change PORT in `.env` to a different port (e.g., 8001), then access at `http://localhost:8001`

### Error: "SECRET_KEY is too short"
**Solution**: Make sure SECRET_KEY in `.env` is at least 32 characters long

### Application starts but shows errors in browser
**Solutions**:
- Check browser console (F12) for JavaScript errors
- Verify all static files are accessible (check `static/js/auth.js` exists)
- Clear browser cache and reload

## What to Expect

✅ **On startup**: Database tables are created automatically  
✅ **In guest mode**: You can view pages but cannot run analysis  
✅ **After login**: You can upload images and run ML analysis  
✅ **ML Analysis**: Real detection from satellite imagery (no fake data)

## Next Steps

- Register and create your first analysis
- Try different locations with satellite imagery
- View analysis history in your account
- Check the API documentation at `http://localhost:8000/docs` (FastAPI auto-generated docs)

