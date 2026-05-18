# Quick Setup Guide

## Step 1: Install PostgreSQL

Download and install PostgreSQL from https://www.postgresql.org/download/

Create a database:
```sql
CREATE DATABASE solar_analyzer;
```

## Step 2: Install Python Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Configure Environment

1. Copy `.env.example` to `.env`:
```bash
copy .env.example .env  # Windows
# or
cp .env.example .env  # Linux/Mac
```

2. Edit `.env` and update:
   - `DATABASE_URL` with your PostgreSQL credentials
   - `SECRET_KEY` with a secure random string (min 32 characters)

## Step 4: Run Application

```bash
python main.py
```

The application will be available at `http://localhost:8000`

## Step 5: Test

1. Open browser to `http://localhost:8000`
2. You should see the landing page in guest mode
3. Click "Register" to create an account
4. After registration, you can run solar analysis

## Troubleshooting

### Database Connection Error
- Ensure PostgreSQL is running
- Check `DATABASE_URL` in `.env`
- Verify database exists

### Import Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

### Port Already in Use
- Change `PORT` in `.env` to a different port (e.g., 8001)

