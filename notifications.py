# notifications.py
import os
import time
import webbrowser
from fpdf import FPDF
from typing import List, Dict, Any, Optional
import requests
import urllib.parse
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

# Logging using centralized system
from logging_config import get_logger
from settings import load_settings  # Import settings to get notification limits

logger = get_logger(__name__)

# Config
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
WHATSAPP_NUMBER = os.getenv("WHATSAPP_TO", "")

# Load settings for notification limits
SETTINGS = load_settings()
DISCORD_SIGNAL_LIMIT = int(SETTINGS.get("DISCORD_SIGNAL_LIMIT", 5))
TELEGRAM_SIGNAL_LIMIT = int(SETTINGS.get("TELEGRAM_SIGNAL_LIMIT", 5))
WHATSAPP_SIGNAL_LIMIT = int(SETTINGS.get("WHATSAPP_SIGNAL_LIMIT", 3))

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
            symbol = s.get('symbol', 'N/A')
            exchange = s.get('exchange', 'N/A').capitalize()
            self.cell(0, 6, f"Signal #{i+1}: {symbol} ({exchange})", ln=1)
            
            # Signal details
            self.set_font("Arial", "", 9)
            self.set_text_color(50, 50, 50)
            
            signal_type = s.get('signal_type', 'N/A')
            side = s.get('side', 'N/A')
            score = s.get('score', 'N/A')
            entry = s.get('entry', 'N/A')
            tp = s.get('tp', 'N/A')
            sl = s.get('sl', 'N/A')
            market = s.get('market', 'N/A')
            trend_score = s.get('indicators', {}).get('trend_score', 'N/A')
            trail = s.get('trail', 'N/A')
            margin = s.get('margin_usdt', 'N/A')
            liq = s.get('liquidation', 'N/A')
            
            created_at = s.get('created_at', datetime.now(timezone.utc))
            if isinstance(created_at, datetime):
                created_at = created_at.strftime('%Y-%m-%d %H:%M:%S %Z')
            else:
                try:
                    created_at = datetime.fromisoformat(str(created_at).replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S %Z')
                except:
                    created_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            details = [
                f"Type: {signal_type} | Side: {side} | Score: {score}%",
                f"Entry: {entry} | TP: {tp} | SL: {sl}",
                f"Market: {market} | Trend Score: {trend_score} | Trail: {trail}",
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
        logger.error(f"Error generating PDF: {e}", exc_info=True)
        return b""

def format_signal_block(signal: Dict[str, Any]) -> str:
    """Format a single signal for human-readable notification (Markdown style)"""
    try:
        symbol = signal.get("symbol", "Unknown")
        exchange = signal.get("exchange", "Unknown").capitalize()
        market = signal.get("market", "Normal")
        side = str(signal.get("side", "Buy")).upper()
        score = float(signal.get("score", 0))
        entry = float(signal.get("entry", 0))
        tp = float(signal.get("tp", 0))
        sl = float(signal.get("sl", 0))
        trail = float(signal.get("trail", 0))
        margin = float(signal.get("margin_usdt", 5))
        liquidation = float(signal.get("liquidation", 0))
        trend_score = float(signal.get("indicators", {}).get("trend_score", 0))

        created_at = signal.get("created_at", datetime.now(timezone.utc))
        if isinstance(created_at, datetime):
            created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            try:
                created_at_str = datetime.fromisoformat(str(created_at).replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M:%S %Z")
            except:
                created_at_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

        return (
            f"💹 **{symbol}** ({exchange}) - {market} Market\n"
            f"🔹 **{side}** Signal | Score: **{score:.1f}%**\n"
            f"🔹 Entry: **${entry:.6f}** | TP: **${tp:.6f}** | SL: **${sl:.6f}**\n"
            f"🔹 Trend Score: {trend_score:.2f} | Trail: ${trail:.6f}\n"
            f"🔹 Margin: ${margin:.6f} | Liquidation: ${liquidation:.6f}\n"
            f"🔹 Generated: {created_at_str}\n"
        )
    except (ValueError, TypeError) as e:
        logger.error(f"Error formatting signal block: {e}", exc_info=True)
        return f"💹 Signal formatting error for {signal.get('symbol', 'Unknown')}: {str(e)}\n"

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
        signal_blocks = [format_signal_block(s) for s in signals[:WHATSAPP_SIGNAL_LIMIT]]
        message_header = f"🚀 AlgoTrader Pro - {len(signals)} Signals Generated\n\n"
        message = message_header + "\n".join(signal_blocks)
        
        if len(signals) > WHATSAPP_SIGNAL_LIMIT:
            message += f"\n\n📊 {len(signals) - WHATSAPP_SIGNAL_LIMIT} more signals available in the app"
        
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
        signal_blocks = [format_signal_block(s) for s in signals[:DISCORD_SIGNAL_LIMIT]]
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
        signal_blocks = [format_signal_block(s) for s in signals[:TELEGRAM_SIGNAL_LIMIT]]
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
        'symbol': 'BTCUSDT',
        'exchange': 'binance',
        'signal_type': 'bullish',
        'side': 'BUY',
        'score': 85.5,
        'entry': 45000.00,
        'tp': 46500.00,
        'sl': 44000.00,
        'market': 'futures',
        'trail': 250.00,
        'margin_usdt': 150.00,
        'liquidation': 40500.00,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'indicators': {'trend_score': 1.0}
    }
    
    logger.info("Testing notification system...")
    send_all_notifications([test_signal])