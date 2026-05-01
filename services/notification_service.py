import logging
import httpx
from typing import Optional, Dict, Any
from utils.config import settings

logger = logging.getLogger("ai-inference.notification")

class NotificationService:
    """
    Service de notification pour SOS-Cameroun.
    Gère l'envoi d'emails d'urgence via l'API Brevo (Sendinblue).
    """

    def __init__(self):
        self.api_key = settings.BREVO_API_KEY
        self.sender_email = settings.BREVO_SENDER_EMAIL
        self.sender_name = "SOS-Cameroun Emergency"
        self.base_url = "https://api.brevo.com/v3/smtp/email"

    async def send_emergency_email(
        self,
        to_email: str,
        subject: str,
        incident_data: Dict[str, Any],
        recipient_name: str = "Autorité de Secours",
        victim_name: str = "Un citoyen",
        is_familiar: bool = False
    ) -> bool:
        """
        Envoie un email d'alerte structuré via Brevo.
        is_familiar: Si True, utilise un ton moins formel pour les proches.
        """
        if not self.api_key:
            logger.error("❌ BREVO_API_KEY non configurée. Impossible d'envoyer l'email.")
            return False

        headers = {
            "accept": "application/json",
            "api-key": self.api_key,
            "content-type": "application/json"
        }

        # Personnalisation du message selon le destinataire
        if is_familiar:
            greeting = f"Bonjour {recipient_name},"
            intro = f"Nous t'informons que <strong>{victim_name}</strong> vient de signaler une urgence via l'application SOS-Cameroun. Voici les détails pour que tu puisses l'aider ou rester informé :"
            footer_note = "Reste calme et essaie de le/la joindre si possible."
        else:
            greeting = f"À l'attention de : <strong>{recipient_name}</strong>,"
            intro = f"Une alerte d'urgence concernant <strong>{victim_name}</strong> a été traitée par notre plateforme. Veuillez trouver ci-dessous les détails complets pour intervention :"
            footer_note = "Cette alerte a été validée par les algorithmes SOS-Cameroun. Une intervention est requise."

        # Construction du corps du message en HTML
        html_content = f"""
        <html>
        <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #1a1a1a; background-color: #f4f4f4; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: auto; background: #ffffff; border: 1px solid #e0e0e0; border-radius: 4px; overflow: hidden;">
                <div style="background-color: {'#d9534f' if is_familiar else '#004a99'}; color: #ffffff; padding: 25px; text-align: center; border-bottom: 4px solid #d9534f;">
                    <h1 style="margin: 0; font-size: 20px; text-transform: uppercase; letter-spacing: 1px;">{ "ALERTE PROCHE" if is_familiar else "Alerte d'Urgence Nationale"}</h1>
                    <p style="margin: 5px 0 0; font-size: 13px; opacity: 0.9;">Système SOS-Cameroun - Centre National d'Urgence</p>
                </div>
                
                <div style="padding: 30px;">
                    <p style="font-size: 15px;">{greeting}</p>
                    <p style="margin-bottom: 20px;">{intro}</p>
                    
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                        <tr style="background-color: #f9f9f9;">
                            <td style="padding: 12px; border: 1px solid #eeeeee; width: 40%;"><strong>Type d'incident</strong></td>
                            <td style="padding: 12px; border: 1px solid #eeeeee; color: #d9534f; font-weight: bold;">{incident_data.get('type_incident', 'Non spécifié')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #eeeeee;"><strong>Degré de gravité</strong></td>
                            <td style="padding: 12px; border: 1px solid #eeeeee;">{incident_data.get('gravite', 'Moyenne')}</td>
                        </tr>
                        <tr style="background-color: #f9f9f9;">
                            <td style="padding: 12px; border: 1px solid #eeeeee;"><strong>Localisation</strong></td>
                            <td style="padding: 12px; border: 1px solid #eeeeee;">{incident_data.get('lieu', 'Localisation en cours')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px; border: 1px solid #eeeeee;"><strong>Coordonnées GPS</strong></td>
                            <td style="padding: 12px; border: 1px solid #eeeeee; font-family: monospace;">{incident_data.get('gps', 'Indisponibles')}</td>
                        </tr>
                        <tr style="background-color: #f9f9f9;">
                            <td style="padding: 12px; border: 1px solid #eeeeee;"><strong>Description</strong></td>
                            <td style="padding: 12px; border: 1px solid #eeeeee;">{incident_data.get('description', 'Aucune description fournie')}</td>
                        </tr>
                    </table>
                    
                    <div style="background-color: #fff9f9; border-left: 4px solid #d9534f; padding: 15px; margin-bottom: 20px;">
                        <p style="margin: 0; font-size: 13px; color: #a94442;">
                            <strong>Note :</strong> {footer_note}
                        </p>
                    </div>
                </div>
                
                <div style="background-color: #f1f1f1; padding: 15px; text-align: center; font-size: 11px; color: #666666; border-top: 1px solid #e0e0e0;">
                    <p style="margin: 0;">Message automatique sécurisé émis par la plateforme SOS-Cameroun.</p>
                    <p style="margin: 5px 0 0;">© 2026 SOS-Cameroun - République du Cameroun.</p>
                </div>
            </div>
        </body>
        </html>
        """

        payload = {
            "sender": {"name": self.sender_name, "email": self.sender_email},
            "to": [{"email": to_email, "name": recipient_name}],
            "subject": subject,
            "htmlContent": html_content
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_url, json=payload, headers=headers)
                if response.status_code in [200, 201]:
                    logger.info(f"✅ Email d'alerte envoyé avec succès à {to_email}")
                    return True
                else:
                    logger.error(f"❌ Erreur Brevo ({response.status_code}) : {response.text}")
                    return False
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'envoi de l'email : {e}")
                return False

notification_service = NotificationService()
