# Deployment Guide - Asinax

This guide provides detailed instructions for deploying the Asinax platform to a production server.

## 1. Server Requirements

- **Operating System**: Ubuntu 22.04 LTS (recommended)
- **CPU**: 2+ vCPUs
- **RAM**: 4GB+ (8GB recommended for optimal performance)
- **Storage**: 40GB+ SSD
- **Networking**: Static public IP address

## 2. Domain and DNS

- **Domain**: You need a registered domain name (e.g., `asinax.com`).
- **DNS `A` Record**: Create an `A` record in your DNS provider's dashboard that points your domain (and the `www` subdomain) to your server's public IP address.

## 3. Installation

The easiest way to deploy the platform is by using the interactive installation script.

### Step 1: Connect to Your Server

Connect to your server via SSH:
```bash
ssh root@YOUR_SERVER_IP
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/hseen199/legendary-ai-trader.git
cd legendary-ai-trader
```

### Step 3: Run the Installation Script

The script will handle everything from installing Docker to configuring SSL.

```bash
chmod +x scripts/install.sh
sudo bash scripts/install.sh
```

**The script will ask for the following:**
1.  **Confirmation to proceed**: Type `y` and press Enter.
2.  **SSL Setup**: Type `y` to set up SSL (recommended).
3.  **Domain Name**: Enter your domain (e.g., `asinax.com`).
4.  **Email Address**: For Let's Encrypt notifications.
5.  **Admin Credentials**: Set the email and password for the main admin account.

### Step 4: Post-Installation Configuration

After the script finishes, you need to add your secret keys to the environment files.

1.  **Edit the main `.env` file:**
    ```bash
    nano .env
    ```
    -   `BINANCE_API_KEY`: Your Binance API key.
    -   `BINANCE_API_SECRET`: Your Binance API secret.
    -   `SMTP_USER`: Your email address for sending notifications.
    -   `SMTP_PASSWORD`: Your email app password.
    -   `OPENAI_API_KEY`: Your OpenAI API key (optional, for future features).

2.  **Edit the backend `.env` file:**
    ```bash
    nano backend/.env
    ```
    -   This file should already be populated with the correct database credentials. Double-check that the `BINANCE_API_KEY`, `BINANCE_API_SECRET`, and SMTP settings are also correct here.

### Step 5: Restart Services

After editing the `.env` files, restart the Docker containers to apply the changes.

```bash
docker-compose restart
```

## 4. Verifying the Installation

-   **Website**: Open your domain in a browser (`https://your-domain.com`). You should see the Asinax landing page.
-   **Admin Panel**: Navigate to `https://your-domain.com/admin` and log in with the credentials you created.
-   **Service Status**: Run the status script to check if all services are healthy.
    ```bash
    sudo bash scripts/status.sh
    ```

## 5. Backups and Maintenance

-   **Create Backups**: Regularly run the backup script.
    ```bash
    sudo bash scripts/backup.sh
    ```
-   **Update the Platform**: To update to the latest version from Git:
    ```bash
    sudo bash scripts/update.sh
    ```
-   **Restore from Backup**: To restore your data:
    ```bash
    sudo bash scripts/restore.sh backups/your_backup_file.tar.gz
    ```

## 6. Firewall

The installation script automatically configures `ufw` (Uncomplicated Firewall) to allow traffic on essential ports:
-   **22 (SSH)**
-   **80 (HTTP)**
-   **443 (HTTPS)**

You can check the firewall status with `sudo ufw status`.
