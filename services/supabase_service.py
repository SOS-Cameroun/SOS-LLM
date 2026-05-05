import logging
import os
from supabase import create_client, Client
from utils.config import settings

logger = logging.getLogger("ai-inference.supabase")

class SupabaseService:
    def __init__(self):
        # Récupération et nettoyage des credentials
        self.url = settings.SUPABASE_URL.strip() if settings.SUPABASE_URL else None
        self.key = settings.SUPABASE_KEY.strip() if settings.SUPABASE_KEY else None
        
        if not self.url or not self.key:
            logger.warning("⚠️ Supabase credentials not configured. Supabase features will be disabled.")
            self.client = None
        else:
            # Diagnostic (masqué pour la sécurité)
            masked_key = f"{self.key[:6]}...{self.key[-4:]}" if len(self.key) > 10 else "***"
            logger.info(f"🔌 Tentative d'initialisation Supabase sur {self.url}")
            logger.debug(f"🔑 Key format: len={len(self.key)}, prefix={self.key[:5]}")
            
            try:
                self.client: Client = create_client(self.url, self.key)
                logger.info("✅ Supabase client initialized.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase client: {e}")
                logger.error(f"👉 Vérifiez que la clé '{masked_key}' est bien une clé 'anon' valide.")
                self.client = None

    def register_citizen(self, nom: str, email: str, telephone: str, contact_email: str, contact_phone: str, nom_contact: str):
        """
        Enregistre un citoyen et son premier contact d'urgence dans Supabase.
        Logique : Citoyen d'abord, puis Contact lié.
        """
        if not self.client:
            raise RuntimeError("Supabase client not initialized")

        try:
            # 1. Insérer le citoyen (avec email et téléphone)
            citizen_data = {
                "nom": nom,
                "email": email,
                "telephone": telephone,
            }
            citizen_res = self.client.table("citoyen").insert(citizen_data).execute()
            
            if not citizen_res.data:
                raise RuntimeError("Failed to insert citizen")
                
            citizen_id = citizen_res.data[0]["id"]
            
            # 2. Insérer le contact d'urgence lié au citoyen
            contact_data = {
                "email": contact_email,
                "telephone": contact_phone,
                "id_citoyen": citizen_id,
                "nom":nom_contact
            }
            contact_res = self.client.table("contact_urgence").insert(contact_data).execute()
            
            if not contact_res.data:
                raise RuntimeError("Failed to insert emergency contact")
            
            return {
                "citizen": citizen_res.data[0],
                "contact": contact_res.data[0]
            }

        except Exception as e:
            logger.error(f"❌ Error during registration: {e}")
            raise e

    def get_citizen_name(self, citizen_id: str):
        """
        Récupère le nom d'un citoyen par son ID.
        """
        if not self.client:
            return "Un citoyen"
        
        try:
            res = self.client.table("citoyen").select("nom").eq("id", citizen_id).execute()
            if res.data:
                return res.data[0]["nom"]
            return "Un citoyen"
        except Exception as e:
            logger.error(f"❌ Error fetching name for citizen {citizen_id}: {e}")
            return "Un citoyen"

    def get_citizen(self, citizen_id: str):
        """
        Récupère les informations complètes d'un citoyen par son ID.
        """
        if not self.client:
            return None
        
        try:
            res = self.client.table("citoyen").select("*").eq("id", citizen_id).execute()
            if res.data:
                return res.data[0]
            return None
        except Exception as e:
            logger.error(f"❌ Error fetching citizen {citizen_id}: {e}")
            return None

    def get_citizen_contacts(self, citizen_id: str):
        """
        Récupère tous les contacts d'urgence d'un citoyen.
        """
        if not self.client:
            return []
        
        try:
            res = self.client.table("contact_urgence").select("*").eq("id_citoyen", citizen_id).execute()
            return res.data or []
        except Exception as e:
            logger.error(f"❌ Error fetching contacts for citizen {citizen_id}: {e}")
            return []

    def add_emergency_contact(self, citizen_id: str, email: str, phone: str, nom: str = "Contact d'Urgence"):
        """
        Ajoute un nouveau contact d'urgence pour un citoyen existant.
        """
        if not self.client:
            raise RuntimeError("Supabase client not initialized")

        try:
            contact_data = {
                "email": email,
                "telephone": phone,
                "id_citoyen": citizen_id,
                "nom": nom
            }
            res = self.client.table("contact_urgence").insert(contact_data).execute()
            
            if not res.data:
                raise RuntimeError("Failed to add emergency contact")
                
            return res.data[0]
        except Exception as e:
            logger.error(f"❌ Error adding contact for citizen {citizen_id}: {e}")
            raise e

    def upload_audio(self, file_path: str, bucket_name: str = "audio"):
        """
        Upload un fichier audio vers un bucket Supabase et retourne son URL publique.
        """
        if not self.client:
            logger.warning("Supabase client not initialized, cannot upload audio.")
            return None

        try:
            file_name = os.path.basename(file_path)
            with open(file_path, 'rb') as f:
                # Upload du fichier
                res = self.client.storage.from_(bucket_name).upload(
                    path=file_name,
                    file=f,
                    file_options={"content-type": "audio/mpeg"}
                )
            
            # Récupération de l'URL publique
            public_url = self.client.storage.from_(bucket_name).get_public_url(file_name)
            logger.info(f"✅ Audio uploadé vers Supabase : {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'upload vers Supabase : {e}")
            return None

supabase_service = SupabaseService()
