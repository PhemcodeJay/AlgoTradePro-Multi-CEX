import os
import time
import webbrowser
from fpdf import FPDF
from typing import List, Dict, Any, Optional
import requests
import urllib.parse
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
load_dotenv()

tz_utc3 = timezone(timedelta(hours=3))

# Logging using centralized system
from logging_config import get_logger
logger = get_logger(__name__)

# Config
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
WHATSAPP_NUMBER = os.getenv("WHATSAPP_TO", "")

# PDF Helper
class SignalPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "AlgoTrader Pro - Trading Signals", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def add_signals(self, signals: List[Dict[str, Any]]):
        self.set_font("Arial", size=9)
        for i, s in enumerate(signals):
            if i > 0:
                self.ln(3)
            
            # Signal header
            self.set_font("Arial", "B", 10)
            self.set_text_color(0, 0, 0)
            symbol = s.get('symbol', s.get('Symbol', 'N/A'))
            self.cell(0, 6, f"Signal #{i+1}: {symbol}", ln=1)
            
            # Signal details
            self.set_font("Arial", "", 9)
            self.set_text_color(50, 50, 50)
            
            signal_type = s.get('signal_type', s.get('Type', 'N/A'))
            side = s.get('side', s.get('Side', 'N/A'))
            score = s.get('score', s.get('Score', 'N/A'))
            entry = s.get('entry', s.get('Entry', 'N/A'))
            tp = s.get('tp', s.get('TP', 'N/A'))
            sl = s.get('sl', s.get('SL', 'N/A'))
            market = s.get('market', s.get('Market', 'N/A'))
            bb_slope = s.get('bb_slope', s.get('BB Slope', 'N/A'))
            trail = s.get('trail', s.get('Trail', 'N/A'))
            margin = s.get('margin_usdt', s.get('Margin', 'N/A'))
            liq = s.get('liquidation', s.get('Liq', 'N/A'))
            
            created_at = s.get('created_at', s.get('Time', 'N/A'))
            if isinstance(created_at, datetime):
                created_at = created_at.strftime('%Y-%m-%d %H:%M:%S')
            
            details = [
                f"Type: {signal_type} | Side: {side} | Score: {score}%",
                f"Entry: {entry} | TP: {tp} | SL: {sl}",
                f"Market: {market} | BB: {bb_slope} | Trail: {trail}",
                f"Margin: {margin} | Liq: {liq} | Time: {created_at}"
            ]
            
            for detail in details:
                self.cell(0, 5, str(detail), ln=1)
            
            # Separator line
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
            self.ln(3)

def generate_pdf_bytes(signals: List[Dict[str, Any]]) -> bytes:
    """Generate PDF from signals and return as bytes"""
    if not signals:
        return b""
    
    try:
        pdf = SignalPDF()
        pdf.add_page()
        pdf.add_signals(signals[:25])  # Limit to 25 signals per PDF
        
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            return pdf_output.encode('latin-1')
        return pdf_output
            
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return b""

def format_signal_block(signal: Dict[str, Any]) -> str:
    """Format a single signal for human-readable notification (Markdown style)"""
    try:
        symbol = signal.get("symbol", signal.get("Symbol", "Unknown"))
        market = signal.get("market", signal.get("Market", "Normal"))
        side = str(signal.get("side", signal.get("Side", "Buy"))).upper()
        score = float(signal.get("score", signal.get("Score", 0)))
        entry = float(signal.get("entry", signal.get("Entry", 0)))
        tp = float(signal.get("tp", signal.get("TP", 0)))
        sl = float(signal.get("sl", signal.get("SL", 0)))
        trail = float(signal.get("trail", signal.get("Trail", 0)))
        margin = float(signal.get("margin_usdt", signal.get("Margin", 5)))
        liquidation = float(signal.get("liquidation", signal.get("Liq", 0)))
        bb_slope = signal.get("bb_slope", signal.get("BB Slope", "Unknown"))

        # Fix: convert datetime to timestamp if necessary
        created_at_val = signal.get("created_at", time.time())
        if isinstance(created_at_val, datetime):
            created_at_val = created_at_val.timestamp()
        elif isinstance(created_at_val, str):
            try:
                created_at_val = datetime.fromisoformat(created_at_val.replace('Z', '+00:00')).timestamp()
            except:
                created_at_val = time.time()

        created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at_val))

        return (
            f"💹 **{symbol}** - {market} Market\n"
            f"🔹 **{side}** Signal | Score: **{score:.1f}%**\n"
            f"🔹 Entry: **${entry:.6f}** | TP: **${tp:.6f}** | SL: **${sl:.6f}**\n"
            f"🔹 BB Slope: {bb_slope} | Trail: ${trail:.6f}\n"
            f"🔹 Margin: ${margin:.6f} | Liquidation: ${liquidation:.6f}\n"
            f"🔹 Generated: {created_at}\n"
        )
    except Exception as e:
        logger.error(f"Error formatting signal block: {e}")
        return f"💹 Signal formatting error: {str(e)}\n"

def send_whatsapp(signals: List[Dict[str, Any]], to_number: Optional[str] = None):
    """Open WhatsApp Web with trading signals ready to send"""
    if not to_number:
        to_number = WHATSAPP_NUMBER
    
    if not to_number:
        logger.info("WhatsApp: No phone number configured, skipping")
        return
    
    if not signals:
        logger.warning("WhatsApp: No signals to send")
        return
    
    try:
        signal_blocks = [format_signal_block(s) for s in signals[:3]]
        message_header = f"🚀 AlgoTrader Pro - {len(signals)} Signals Generated\n\n"
        message = message_header + "\n".join(signal_blocks)
        
        if len(signals) > 3:
            message += f"\n\n📊 {len(signals) - 3} more signals available in the app"
        
        # Clean up message for WhatsApp (remove markdown)
        message = message.replace("**", "")
        
        encoded_message = urllib.parse.quote(message)
        whatsapp_url = f"https://wa.me/{to_number}?text={encoded_message}"
        webbrowser.open(whatsapp_url)
        logger.info(f"WhatsApp message prepared for {to_number}")
        
    except Exception as e:
        logger.error(f"WhatsApp error: {e}", exc_info=True)

def send_discord(signals: List[Dict[str, Any]]):
    """Send signals to Discord webhook"""
    if not DISCORD_WEBHOOK_URL:
        logger.info("Discord: No webhook URL configured, skipping")
        return
    
    if not signals:
        logger.warning("Discord: No signals to send")
        return
    
    try:
        signal_blocks = [format_signal_block(s) for s in signals[:5]]
        message = f"🎯 **AlgoTrader Pro - Top {len(signal_blocks)} Trading Signals**\n\n" + "\n".join(signal_blocks)
        
        # Discord has 2000 char limit
        if len(message) > 1900:
            message = message[:1900] + "...\n\n📊 **Full report in attached PDF**"
        
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
        response.raise_for_status()
        logger.info("Discord message sent successfully")
        
        # Send PDF attachment
        pdf_bytes = generate_pdf_bytes(signals)
        if pdf_bytes and len(pdf_bytes) > 0:
            files = {'file': ('trading_signals.pdf', pdf_bytes, 'application/pdf')}
            pdf_response = requests.post(DISCORD_WEBHOOK_URL, files=files, timeout=15)
            pdf_response.raise_for_status()
            logger.info("Discord PDF attachment sent successfully")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Discord HTTP error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Discord error: {e}", exc_info=True)

def send_telegram(signals: List[Dict[str, Any]]):
    """Send signals to Telegram bot"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram: Missing credentials, skipping")
        return
    
    if not signals:
        logger.warning("Telegram: No signals to send")
        return
    
    try:
        signal_blocks = [format_signal_block(s) for s in signals[:5]]
        message = f"🎯 *AlgoTrader Pro - Top {len(signal_blocks)} Trading Signals*\n\n" + "\n".join(signal_blocks)
        
        # Telegram has 4096 char limit
        if len(message) > 4000:
            message = message[:4000] + "...\n\n📊 *Full report in PDF attachment*"
        
        send_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        response = requests.post(send_url, data={
            "chat_id": TELEGRAM_CHAT_ID, 
            "text": message, 
            "parse_mode": "Markdown"
        }, timeout=10)
        response.raise_for_status()
        logger.info("Telegram message sent successfully")
        
        # Send PDF attachment
        pdf_bytes = generate_pdf_bytes(signals)
        if pdf_bytes and len(pdf_bytes) > 0:
            doc_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
            files = {'document': ('trading_signals.pdf', pdf_bytes, 'application/pdf')}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": f"AlgoTrader Pro - {len(signals)} Trading Signals"}
            pdf_response = requests.post(doc_url, data=data, files=files, timeout=15)
            pdf_response.raise_for_status()
            logger.info("Telegram PDF attachment sent successfully")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Telegram HTTP error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Telegram error: {e}", exc_info=True)

def send_all_notifications(signals: List[Dict[str, Any]]):
    """Send signals to all configured notification channels"""
    if not signals:
        logger.warning("No signals to send")
        return
    
    logger.info(f"Sending {len(signals)} signals to notification channels")
    
    try:
        send_discord(signals)
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
    
    try:
        send_telegram(signals)
    except Exception as e:
        logger.error(f"Telegram notification failed: {e}")
    
    try:
        send_whatsapp(signals)
    except Exception as e:
        logger.error(f"WhatsApp notification failed: {e}")
    
    logger.info("Notification broadcast completed")

def test_notifications():
    """Test notification system with sample signal"""
    test_signal = {
        'Symbol': 'BTCUSDT',
        'Type': 'Buy',
        'Side': 'LONG',
        'Score': '85.5',
        'Entry': 45000.00,
        'TP': 46500.00,
        'SL': 44000.00,
        'Market': 'High Vol',
        'BB Slope': 'Expanding',
        'Trail': 250.00,
        'Margin': 150.00,
        'Liq': 40500.00,
        'Time': '2024-01-15 10:30:00'
    }
    
    logger.info("Testing notification system...")
    send_all_notifications([test_signal])
