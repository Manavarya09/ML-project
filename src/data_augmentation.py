"""
Data augmentation script to generate synthetic phishing and ham emails.
"""

import random
import pandas as pd
from pathlib import Path
from src.config import config
from src.utils import LOGGER

# Templates for Phishing Emails
PHISHING_TEMPLATES = [
    "URGENT: Your account at {bank} has been suspended due to suspicious activity. Verify your identity immediately at {link} to avoid permanent closure.",
    "Dear User, your payment of {amount} to {company} was successful. If you did not authorize this, click here {link} to cancel immediately.",
    "HR Update: Please review the attached policy changes regarding your benefits. Login to the portal here: {link}",
    "CONGRATULATIONS! You have won a {prize} in our yearly raffle. Claim your prize now by filling out this form: {link}",
    "Security Alert: A new device signed in to your {service} account from {location}. If this wasn't you, secure your account now: {link}",
    "Final Notice: Your invoice #{invoice_id} is overdue. Pay now to avoid service interruption. View invoice: {link}",
    "CEO Request: I need you to purchase {gift_cards} gift cards for a client immediately. I am in a meeting, so please handle this discreetly. Reply with the codes.",
    "IT Support: Your password for {system} expires in 24 hours. Update it now to maintain access: {link}",
    "Verify your email address to complete your registration for {app}. Click here: {link}",
    "Your package from {courier} is on hold due to unpaid customs fees. Pay {fee} to release delivery: {link}"
]

# Templates for Legitimate (Ham) Emails
HAM_TEMPLATES = [
    "Hi team, just a reminder that our weekly sync is tomorrow at {time}. Please update the agenda doc.",
    "Attached is the project report for {project}. Let me know if you have any feedback by EOD.",
    "Can we reschedule our 1:1 to {day}? Something urgent came up.",
    "Here are the minutes from today's meeting. Action items are highlighted in bold.",
    "Happy Birthday {name}! Hope you have a fantastic day.",
    "The server maintenance is scheduled for {day} night. Expect some downtime.",
    "Please find the requested documents attached. Best regards, {name}.",
    "Are you free for a quick call to discuss the {project} timeline?",
    "Thanks for the update. I'll review and get back to you shortly.",
    "Invitation: Lunch and Learn session on {topic} this Friday in the main conference room."
]

# Slot fillers
BANKS = ["Chase", "Wells Fargo", "Bank of America", "Citi", "PayPal"]
COMPANIES = ["Amazon", "Apple", "Netflix", "Microsoft", "Google"]
LINKS = ["http://bit.ly/secure-login", "http://verify-account-now.com", "http://secure-update.net", "http://login-support-portal.info"]
PRIZES = ["iPhone 15", "$1000 Gift Card", "Luxury Vacation", "Tesla Model 3"]
SERVICES = ["Netflix", "Spotify", "Dropbox", "Slack", "Zoom"]
LOCATIONS = ["Russia", "China", "Nigeria", "Unknown IP"]
COURIERS = ["FedEx", "UPS", "DHL", "USPS"]
TIMES = ["10:00 AM", "2:00 PM", "11:30 AM", "4:00 PM"]
PROJECTS = ["Alpha", "Phoenix", "Apollo", "Gemini", "Orion"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
TOPICS = ["AI Safety", "Cloud Computing", "Cybersecurity", "Data Science"]
GIFT_CARDS = ["Apple", "Google Play", "Amazon", "Steam"]

def generate_synthetic_data(num_samples=500):
    """Generate synthetic emails and append to dataset."""
    new_data = []
    
    for _ in range(num_samples):
        is_phishing = random.choice([True, False])
        
        if is_phishing:
            template = random.choice(PHISHING_TEMPLATES)
            text = template.format(
                bank=random.choice(BANKS),
                link=random.choice(LINKS),
                amount=f"${random.randint(50, 5000)}",
                company=random.choice(COMPANIES),
                prize=random.choice(PRIZES),
                service=random.choice(SERVICES),
                location=random.choice(LOCATIONS),
                invoice_id=random.randint(10000, 99999),
                gift_cards=random.choice(GIFT_CARDS),
                system="Office 365",
                app="Zoom",
                courier=random.choice(COURIERS),
                fee=f"${random.randint(10, 50)}"
            )
            label = 1
        else:
            template = random.choice(HAM_TEMPLATES)
            text = template.format(
                time=random.choice(TIMES),
                project=random.choice(PROJECTS),
                day=random.choice(DAYS),
                name=random.choice(NAMES),
                topic=random.choice(TOPICS)
            )
            label = 0
            
        new_data.append({"email_text": text, "label": label})
    
    return pd.DataFrame(new_data)

def main():
    LOGGER.info("Generating synthetic data...")
    df_new = generate_synthetic_data(num_samples=600) # Generate 600 new samples
    
    data_path = config.data_path
    if data_path.exists():
        df_old = pd.read_csv(data_path)
        LOGGER.info(f"Original dataset size: {len(df_old)}")
        
        # Ensure columns match (handle text vs email_text)
        if "text" in df_old.columns and "email_text" not in df_old.columns:
             df_new = df_new.rename(columns={"email_text": "text"})
        
        # Concatenate
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        
        # Save back
        df_combined.to_csv(data_path, index=False)
        LOGGER.info(f"New dataset size: {len(df_combined)}")
        LOGGER.info(f"Appended {len(df_new)} synthetic samples to {data_path}")
    else:
        LOGGER.error(f"Dataset not found at {data_path}")

if __name__ == "__main__":
    main()
