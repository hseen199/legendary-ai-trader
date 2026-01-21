# Asinax - Legendary AI Trading Platform

**Asinax** is a comprehensive, production-ready AI trading platform designed for automated cryptocurrency investment. It features a robust FastAPI backend, a modern React frontend, and a sophisticated AI trading bot.

---

## 🚀 Features

- **User Management**: Secure registration, login, and profile management.
- **NAV-Based Accounting**: Profits are calculated based on Net Asset Value (NAV).
- **Automated Deposits**: Seamless crypto deposits via NOWPayments (Solana & BEP20).
- **Manual Withdrawals**: Admin-approved withdrawal requests for enhanced security.
- **Referral System**: Reward users for bringing in new investors.
- **VIP Tiers**: Multi-level VIP system (Bronze, Silver, Gold, Platinum).
- **Support System**: Integrated ticketing system for user support.
- **Email Notifications**: Automated email alerts for key events.
- **Advanced Admin Panel**: Comprehensive dashboard for managing users, withdrawals, and platform stats.
- **AI Trading Bot**: Multi-agent AI bot for automated trading on Binance.
- **Dockerized Environment**: Fully containerized for easy deployment and scaling.
- **SSL Support**: Automated SSL certificate generation and renewal with Let's Encrypt.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Backend** | FastAPI, Python 3.11, SQLAlchemy, Pydantic |
| **Frontend** | React, TypeScript, Vite, TailwindCSS, shadcn/ui |
| **Database** | PostgreSQL |
| **Cache** | Redis |
| **Deployment** | Docker, Docker Compose, Nginx |
| **Payments** | NOWPayments |
| **Trading** | Binance API |

---

## 📦 Project Structure

```
legendary-ai-trader/
├── backend/         # FastAPI Backend
├── frontend/        # React Frontend
├── bot/             # AI Trading Bot
├── nginx/           # Nginx Configurations
├── certbot/         # SSL Certificates
├── scripts/         # Management Scripts (install, backup, etc.)
├── .env.example     # Main Environment Variables
├── docker-compose.yml # Docker Compose Configuration
└── README.md        # This File
```

---

## ⚙️ Installation

### Prerequisites

- A server with **Ubuntu 22.04** (or another modern Linux distro).
- Minimum **4GB RAM**, **2 vCPUs**, and **40GB SSD**.
- A registered **domain name** (e.g., `asinax.com`).
- DNS `A` record pointing your domain to the server's IP address.

### Step-by-Step Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hseen199/legendary-ai-trader.git
   cd legendary-ai-trader
   ```

2. **Run the installation script:**
   The interactive script will guide you through the process, including Docker installation, environment setup, and SSL configuration.
   ```bash
   sudo bash scripts/install.sh
   ```

3. **Follow the prompts:**
   - Confirm the installation.
   - Enter your domain name and email for SSL.
   - Set your admin user credentials.

4. **Update API Keys:**
   After installation, edit the `.env` and `backend/.env` files to add your **Binance API keys** and **SMTP credentials**.
   ```bash
   nano .env
   nano backend/.env
   ```

5. **Restart services:**
   ```bash
   docker-compose restart
   ```

Your platform is now live! 🚀

---

## 🔧 Management Scripts

The `scripts/` directory contains several useful scripts for managing your platform:

- `install.sh`: The main installation script.
- `status.sh`: Check the status of all services.
- `backup.sh`: Create a full backup of the database and configuration.
- `restore.sh`: Restore from a backup.
- `update.sh`: Pull the latest code from Git and redeploy.
- `setup-ssl.sh`: Set up or renew SSL certificates.

**Usage:**
```bash
# Check status
sudo bash scripts/status.sh

# Create a backup
sudo bash scripts/backup.sh
```

---

## 💰 Deposit & Withdrawal Flow

- **Deposits**: Users can deposit `SOL` or `BNB` (BEP20). NOWPayments handles the transaction and sends a webhook to the backend. The backend verifies the transaction and credits the user's account with NAV units.
- **Withdrawals**: Users request a withdrawal. An admin must approve the request from the admin panel. Once approved, the withdrawal is processed manually.

---

## ⚠️ Important Notes

- **Security**: Always use strong, unique passwords. Keep your `.env` files secure and do not commit them to public repositories.
- **Backups**: Regularly run the `backup.sh` script and store backups in a secure, off-site location.
- **Binance API**: Ensure your Binance API keys have the correct permissions (trade, read info) and are restricted to your server's IP address for security.
- **2FA**: Two-Factor Authentication is not implemented in this version as per the initial request.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
