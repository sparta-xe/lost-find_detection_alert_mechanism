"""
Professional alert system for lost item detection events.
Supports email, SMS, webhook, and other notification methods.
"""
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Alert configuration."""
    enabled: bool = True
    alert_types: List[str] = None  # ['email', 'webhook', 'sms']
    email_config: Dict = None
    webhook_config: Dict = None
    sms_config: Dict = None
    alert_levels: List[str] = None  # ['info', 'warning', 'critical']

@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    alert_type: str  # 'lost_item_found', 'pickup_attempt', 'suspicious_behavior'
    level: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    timestamp: datetime
    camera_id: str
    additional_data: Dict = None
    image_path: Optional[str] = None

class AlertHandler(ABC):
    """Abstract base class for alert handlers."""
    
    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert."""
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        """Check if handler is properly configured."""
        pass

class EmailAlertHandler(AlertHandler):
    """Email alert handler using SMTP."""
    
    def __init__(self, config: Dict):
        """
        Initialize email handler.
        
        Args:
            config: Email configuration dictionary
        """
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email', self.username)
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
        
        if not isinstance(self.to_emails, list):
            self.to_emails = [self.to_emails]
    
    def is_configured(self) -> bool:
        """Check if email handler is properly configured."""
        return bool(self.username and self.password and self.to_emails)
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send email alert.
        
        Args:
            alert: Alert to send
            
        Returns:
            Success status
        """
        if not self.is_configured():
            logger.error("Email handler not properly configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.level.upper()}] {alert.title}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            
            # Create HTML content
            html_content = self._create_html_content(alert)
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Create plain text content
            text_content = self._create_text_content(alert)
            text_part = MIMEText(text_content, 'plain')
            msg.attach(text_part)
            
            # Attach image if available
            if alert.image_path and Path(alert.image_path).exists():
                with open(alert.image_path, 'rb') as f:
                    img_data = f.read()
                
                image = MIMEImage(img_data)
                image.add_header('Content-Disposition', 'attachment', filename='detection.jpg')
                msg.attach(image)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert {alert.alert_id}: {e}")
            return False
    
    def _create_html_content(self, alert: Alert) -> str:
        """Create HTML email content."""
        level_colors = {
            'info': '#17a2b8',
            'warning': '#ffc107',
            'critical': '#dc3545'
        }
        
        color = level_colors.get(alert.level, '#6c757d')
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                    <h1 style="margin: 0; font-size: 24px;">🔍 Lost Item Detection Alert</h1>
                    <p style="margin: 10px 0 0 0; font-size: 16px; opacity: 0.9;">{alert.level.upper()} LEVEL</p>
                </div>
                
                <div style="padding: 30px;">
                    <h2 style="color: #333; margin-top: 0;">{alert.title}</h2>
                    
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <p style="margin: 0; color: #555; line-height: 1.6;">{alert.message}</p>
                    </div>
                    
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold; color: #555;">Timestamp:</td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; color: #333;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold; color: #555;">Camera ID:</td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; color: #333;">{alert.camera_id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold; color: #555;">Alert ID:</td>
                            <td style="padding: 8px 0; border-bottom: 1px solid #eee; color: #333;">{alert.alert_id}</td>
                        </tr>
                    </table>
                    
                    {self._format_additional_data_html(alert.additional_data) if alert.additional_data else ''}
                </div>
                
                <div style="background-color: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #eee;">
                    <p style="margin: 0; color: #6c757d; font-size: 14px;">
                        Lost Item Detection System | Automated Alert
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_text_content(self, alert: Alert) -> str:
        """Create plain text email content."""
        content = f"""
LOST ITEM DETECTION ALERT - {alert.level.upper()} LEVEL

{alert.title}

{alert.message}

Details:
- Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Camera ID: {alert.camera_id}
- Alert ID: {alert.alert_id}
"""
        
        if alert.additional_data:
            content += "\nAdditional Information:\n"
            for key, value in alert.additional_data.items():
                content += f"- {key}: {value}\n"
        
        content += "\n---\nLost Item Detection System | Automated Alert"
        
        return content
    
    def _format_additional_data_html(self, data: Dict) -> str:
        """Format additional data as HTML table."""
        if not data:
            return ""
        
        html = """
        <div style="margin: 20px 0;">
            <h3 style="color: #333; margin-bottom: 10px;">Additional Information</h3>
            <table style="width: 100%; border-collapse: collapse;">
        """
        
        for key, value in data.items():
            html += f"""
                <tr>
                    <td style="padding: 8px 0; border-bottom: 1px solid #eee; font-weight: bold; color: #555;">{key}:</td>
                    <td style="padding: 8px 0; border-bottom: 1px solid #eee; color: #333;">{value}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
        
        return html

class WebhookAlertHandler(AlertHandler):
    """Webhook alert handler for integrations."""
    
    def __init__(self, config: Dict):
        """
        Initialize webhook handler.
        
        Args:
            config: Webhook configuration dictionary
        """
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.timeout = config.get('timeout', 30)
        self.retry_count = config.get('retry_count', 3)
    
    def is_configured(self) -> bool:
        """Check if webhook handler is properly configured."""
        return bool(self.webhook_url)
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send webhook alert.
        
        Args:
            alert: Alert to send
            
        Returns:
            Success status
        """
        if not self.is_configured():
            logger.error("Webhook handler not properly configured")
            return False
        
        try:
            # Prepare payload
            payload = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'level': alert.level,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'camera_id': alert.camera_id,
                'additional_data': alert.additional_data or {}
            }
            
            # Send webhook with retries
            for attempt in range(self.retry_count):
                try:
                    response = requests.post(
                        self.webhook_url,
                        json=payload,
                        headers=self.headers,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"Webhook alert sent: {alert.alert_id}")
                        return True
                    else:
                        logger.warning(f"Webhook returned status {response.status_code}: {response.text}")
                        
                except requests.RequestException as e:
                    logger.warning(f"Webhook attempt {attempt + 1} failed: {e}")
                    if attempt == self.retry_count - 1:
                        raise
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert {alert.alert_id}: {e}")
            return False

class TelegramAlertHandler(AlertHandler):
    """Telegram bot alert handler."""
    
    def __init__(self, config: Dict):
        """
        Initialize Telegram handler.
        
        Args:
            config: Telegram configuration dictionary
        """
        self.bot_token = config.get('bot_token')
        self.chat_ids = config.get('chat_ids', [])
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if not isinstance(self.chat_ids, list):
            self.chat_ids = [self.chat_ids]
    
    def is_configured(self) -> bool:
        """Check if Telegram handler is properly configured."""
        return bool(self.bot_token and self.chat_ids)
    
    def send_alert(self, alert: Alert) -> bool:
        """
        Send Telegram alert.
        
        Args:
            alert: Alert to send
            
        Returns:
            Success status
        """
        if not self.is_configured():
            logger.error("Telegram handler not properly configured")
            return False
        
        try:
            # Format message
            level_emojis = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'critical': '🚨'
            }
            
            emoji = level_emojis.get(alert.level, '📢')
            
            message = f"""
{emoji} *{alert.title}*

{alert.message}

📅 *Time:* {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
📹 *Camera:* {alert.camera_id}
🆔 *Alert ID:* {alert.alert_id}
"""
            
            if alert.additional_data:
                message += "\n*Additional Info:*\n"
                for key, value in alert.additional_data.items():
                    message += f"• {key}: {value}\n"
            
            # Send to all chat IDs
            success_count = 0
            for chat_id in self.chat_ids:
                try:
                    response = requests.post(
                        f"{self.api_url}/sendMessage",
                        json={
                            'chat_id': chat_id,
                            'text': message,
                            'parse_mode': 'Markdown'
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        logger.warning(f"Telegram API error for chat {chat_id}: {response.text}")
                        
                except Exception as e:
                    logger.error(f"Failed to send Telegram message to chat {chat_id}: {e}")
            
            if success_count > 0:
                logger.info(f"Telegram alert sent to {success_count}/{len(self.chat_ids)} chats: {alert.alert_id}")
                return True
            else:
                logger.error(f"Failed to send Telegram alert to any chat: {alert.alert_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Telegram alert {alert.alert_id}: {e}")
            return False

class AlertManager:
    """Professional alert management system."""
    
    def __init__(self, config: AlertConfig):
        """
        Initialize alert manager.
        
        Args:
            config: Alert configuration
        """
        self.config = config
        self.handlers: Dict[str, AlertHandler] = {}
        
        # Initialize handlers based on configuration
        if config.enabled:
            self._initialize_handlers()
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.max_history = 1000
    
    def _initialize_handlers(self):
        """Initialize alert handlers based on configuration."""
        alert_types = self.config.alert_types or []
        
        # Email handler
        if 'email' in alert_types and self.config.email_config:
            email_handler = EmailAlertHandler(self.config.email_config)
            if email_handler.is_configured():
                self.handlers['email'] = email_handler
                logger.info("Email alert handler initialized")
            else:
                logger.warning("Email handler not properly configured")
        
        # Webhook handler
        if 'webhook' in alert_types and self.config.webhook_config:
            webhook_handler = WebhookAlertHandler(self.config.webhook_config)
            if webhook_handler.is_configured():
                self.handlers['webhook'] = webhook_handler
                logger.info("Webhook alert handler initialized")
            else:
                logger.warning("Webhook handler not properly configured")
        
        # Telegram handler
        if 'telegram' in alert_types and self.config.sms_config:
            telegram_handler = TelegramAlertHandler(self.config.sms_config)
            if telegram_handler.is_configured():
                self.handlers['telegram'] = telegram_handler
                logger.info("Telegram alert handler initialized")
            else:
                logger.warning("Telegram handler not properly configured")
    
    def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """
        Send alert through all configured handlers.
        
        Args:
            alert: Alert to send
            
        Returns:
            Dictionary of handler results
        """
        if not self.config.enabled:
            logger.debug(f"Alerts disabled, skipping alert: {alert.alert_id}")
            return {}
        
        # Check if alert level is enabled
        if self.config.alert_levels and alert.level not in self.config.alert_levels:
            logger.debug(f"Alert level {alert.level} not enabled, skipping: {alert.alert_id}")
            return {}
        
        results = {}
        
        # Send through all handlers
        for handler_name, handler in self.handlers.items():
            try:
                success = handler.send_alert(alert)
                results[handler_name] = success
                
                if success:
                    logger.info(f"Alert sent via {handler_name}: {alert.alert_id}")
                else:
                    logger.error(f"Failed to send alert via {handler_name}: {alert.alert_id}")
                    
            except Exception as e:
                logger.error(f"Error sending alert via {handler_name}: {e}")
                results[handler_name] = False
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        return results
    
    def create_lost_item_alert(self, item_name: str, confidence: float, 
                             camera_id: str, additional_data: Dict = None) -> Alert:
        """Create a lost item found alert."""
        alert_id = f"lost_item_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type='lost_item_found',
            level='warning',
            title=f'Lost Item Found: {item_name}',
            message=f'A lost item "{item_name}" has been detected with {confidence:.1%} confidence.',
            timestamp=datetime.now(),
            camera_id=camera_id,
            additional_data=additional_data
        )
    
    def create_pickup_alert(self, object_name: str, person_id: str, 
                          camera_id: str, additional_data: Dict = None) -> Alert:
        """Create a pickup attempt alert."""
        alert_id = f"pickup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return Alert(
            alert_id=alert_id,
            alert_type='pickup_attempt',
            level='critical',
            title=f'Pickup Attempt Detected',
            message=f'Person {person_id} is attempting to pick up {object_name}.',
            timestamp=datetime.now(),
            camera_id=camera_id,
            additional_data=additional_data
        )
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get recent alert history."""
        return self.alert_history[-limit:]

# Convenience functions
def send_email_alert(message: str, config: Dict = None):
    """Simple email alert function for backward compatibility."""
    if not config:
        logger.warning("No email configuration provided")
        return False
    
    alert_config = AlertConfig(
        enabled=True,
        alert_types=['email'],
        email_config=config
    )
    
    manager = AlertManager(alert_config)
    
    alert = Alert(
        alert_id=f"simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        alert_type='general',
        level='info',
        title='System Alert',
        message=message,
        timestamp=datetime.now(),
        camera_id='unknown'
    )
    
    return manager.send_alert(alert)