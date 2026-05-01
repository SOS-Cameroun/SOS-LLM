import logging
from supabase import create_client, Client
from utils.config import settings

logger = logging.getLogger("ai-inference.supabase")

class SupabaseService:
    def __init__(self):
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_KEY
        
        if not self.url or not self.key:
            logger.warning("⚠️ Supabase credentials not configured. Supabase features will be disabled.")
            self.client = None
        else:
            try:
                self.client: Client = create_client(self.url, self.key)
                logger.info("✅ Supabase client initialized.")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase client: {e}")
                self.client = None

    def register_citizen(self, nom: str, contact_email: str, contact_phone: str, nom_contact:str):
        """
        Enregistre un citoyen et son premier contact d'urgence dans Supabase.
        Logique : Citoyen d'abord, puis Contact lié.
        """
        if not self.client:
            raise RuntimeError("Supabase client not initialized")

        try:
            # 1. Insérer le citoyen
            citizen_data = {"nom": nom}
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

supabase_service = SupabaseService()
