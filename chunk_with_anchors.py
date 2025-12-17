"""
AURORA RAG CHUNKER - DETERMINISTIC DATA TRANSFORMATION
v1.1 - Enhanced Anchors for High Retrieval Recall
"""

import sys
import json

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas openpyxl", file=sys.stderr)
    sys.exit(1)


class AuroraChunker:

    def __init__(self, excel_file):
        try:
            self.sheets = pd.read_excel(excel_file, sheet_name=None)
        except Exception as e:
            print(f"ERROR: Could not read Excel file: {e}", file=sys.stderr)
            sys.exit(1)

        self.chunks = []

        # Load sheets strictly
        self.events_master = self._get_sheet('Events_Master')
        self.event_details = self._get_sheet('Event_Details')
        self.faqs = self._get_sheet('FAQs')
        self.venues = self._get_sheet('Venues')
        self.live_flags = self._get_sheet('Live_Flags')
        self.contacts = self._get_sheet('Contacts')
        self.campus_info = self._get_sheet('Campus_Info')

    def _get_sheet(self, name):
        df = self.sheets.get(name, pd.DataFrame())
        if df.empty:
            return df
        return df.dropna(how='all')

    def _get(self, row, key, default=''):
        try:
            val = row.get(key, default)
            if pd.isna(val):
                return default
            return str(val).strip()
        except:
            return default

    def _event_meta(self, event_id):
        """Returns (event_name, event_type)"""
        row = self.events_master[self.events_master['event_id'] == event_id]
        if row.empty:
            return "", ""
        return self._get(row.iloc[0], 'event_name'), self._get(row.iloc[0], 'event_type')

    def generate_all_chunks(self):
        self._chunk_timing()
        self._chunk_overview()
        self._chunk_topics_prerequisites()
        self._chunk_faqs()
        self._chunk_venues()
        self._chunk_contacts()
        self._chunk_campus_info()
        self._chunk_live_flags()
        return self.chunks

    # ===================== TIMING =====================
    def _chunk_timing(self):
        if self.event_details.empty: return

        for _, day in self.event_details.iterrows():
            event_id = self._get(day, 'event_id')
            venue_id = self._get(day, 'venue_id')
            date = self._get(day, 'date').split()[0]
            start = self._get(day, 'start_time')
            end = self._get(day, 'end_time')

            if not all([event_id, venue_id, date]): continue

            name, etype = self._event_meta(event_id)
            venue_row = self.venues[self.venues['venue_id'] == venue_id]
            if venue_row.empty: continue
            
            v_name = self._get(venue_row.iloc[0], 'venue_name')
            build = self._get(venue_row.iloc[0], 'building')

            # FIX: Added {etype} (e.g. "Hackathon") to anchor
            anchor = f"When is the {name} {etype}? What is the schedule?"
            fact = f"{name} ({etype}) is on {date} from {start} to {end} at {v_name}, {build}."

            self.chunks.append({
                "text": f"{anchor}\n{fact}",
                "metadata": {
                    "chunk_type": "timing",
                    "event_id": event_id,
                    "source": ["event_details", "venues"],
                    "confidence": 1.0
                }
            })

    # ===================== OVERVIEW =====================
    def _chunk_overview(self):
        for _, e in self.events_master.iterrows():
            eid = self._get(e, 'event_id')
            name = self._get(e, 'event_name')
            etype = self._get(e, 'event_type')
            
            if not name: continue

            # FIX: Added variations to anchor
            anchor = f"Is registration required for {name}? Overview of {name} {etype}."
            fact = f"{name} is a {etype} by {self._get(e,'club_name')}. Registration: {self._get(e,'registration_required')}."

            self.chunks.append({
                "text": f"{anchor}\n{fact}",
                "metadata": {"chunk_type": "overview", "event_id": eid, "source": ["events_master"], "confidence": 1.0}
            })

    # ===================== TOPICS / PREREQS =====================
    def _chunk_topics_prerequisites(self):
        for event_id in self.event_details['event_id'].unique():
            row = self.event_details[self.event_details['event_id'] == event_id].iloc[0]
            name, etype = self._event_meta(event_id)

            topics = self._get(row, 'topics_covered')
            prereq = self._get(row, 'prerequisites')
            
            if topics and topics.lower() != 'none':
                self.chunks.append({
                    "text": f"What topics are covered in {name} {etype}?\n{name} covers: {topics}.",
                    "metadata": {"chunk_type": "topics", "event_id": event_id, "source": ["event_details"], "confidence": 1.0}
                })

            if prereq and prereq.lower() != 'none':
                self.chunks.append({
                    "text": f"What are the prerequisites for {name} {etype}?\nPrerequisites: {prereq}.",
                    "metadata": {"chunk_type": "prerequisites", "event_id": event_id, "source": ["event_details"], "confidence": 1.0}
                })

    # ===================== FAQ =====================
    def _chunk_faqs(self):
        for _, f in self.faqs.iterrows():
            q = self._get(f, 'question')
            a = self._get(f, 'answer')
            if not q: continue

            # FAQ is already a question, no extra anchor needed, just semantic boost
            eid = self._get(f, 'event_id')
            name, _ = self._event_meta(eid) if eid else ("", "")
            context = f"regarding {name}" if name else ""

            self.chunks.append({
                "text": f"Q: {q}\nA: {a}", # Keep clean for FAQ
                "metadata": {"chunk_type": "faq", "faq_id": self._get(f, 'faq_id'), "source": ["faqs"], "confidence": 1.0}
            })

    # ===================== VENUES =====================
    def _chunk_venues(self):
        for _, v in self.venues.iterrows():
            name = self._get(v, 'venue_name')
            if not name: continue
            
            # FIX: Added "Location" keyword
            anchor = f"Where is {name}? Location of {name}."
            fact = f"{name} is in {self._get(v,'building')}. Facilities: {self._get(v,'facilities')}."
            
            self.chunks.append({
                "text": f"{anchor}\n{fact}",
                "metadata": {"chunk_type": "venue", "venue_id": self._get(v, 'venue_id'), "source": ["venues"], "confidence": 1.0}
            })

    # ===================== CONTACTS =====================
    def _chunk_contacts(self):
        for _, c in self.contacts.iterrows():
            eid = self._get(c, 'event_id')
            name = self._get(c, 'name')
            if not name: continue

            ename, etype = self._event_meta(eid) if eid != 'ALL' else ("Aurora Fest", "General")
            
            anchor = f"Who is the contact for {ename} {etype}?"
            fact = f"Contact {name} ({self._get(c,'role')}). Phone: {self._get(c,'phone')}."

            self.chunks.append({
                "text": f"{anchor}\n{fact}",
                "metadata": {"chunk_type": "contact", "event_id": eid, "source": ["contacts"], "confidence": 1.0}
            })

    # ===================== CAMPUS =====================
    def _chunk_campus_info(self):
        for _, i in self.campus_info.iterrows():
            title = self._get(i, 'title')
            cat = self._get(i, 'category')
            if not title: continue

            # FIX: Added category to anchor (e.g. "Where is the Food Court (Food)?")
            anchor = f"Where is the {title}? {cat} info."
            fact = f"{title}: {self._get(i,'description')}. Location: {self._get(i,'location')}."

            self.chunks.append({
                "text": f"{anchor}\n{fact}",
                "metadata": {"chunk_type": "campus_info", "source": ["campus_info"], "confidence": 1.0}
            })

    # ===================== LIVE FLAGS =====================
    def _chunk_live_flags(self):
        for _, f in self.live_flags.iterrows():
            if self._get(f, 'is_active').upper() != 'YES': continue
            
            eid = self._get(f, 'event_id')
            ename, _ = self._event_meta(eid) if eid != 'ALL' else ("Aurora Fest", "")
            
            self.chunks.append({
                "text": f"URGENT UPDATE for {ename}: {self._get(f,'message')}",
                "metadata": {"chunk_type": "live_flag", "event_id": eid, "confidence": 1.0}
            })

    def to_json(self):
        return json.dumps(self.chunks, indent=2, ensure_ascii=False)

def main():
    if len(sys.argv) < 2:
        print("Usage: python chunk.py <excel_file>", file=sys.stderr)
        sys.exit(1)
    
    chunker = AuroraChunker(sys.argv[1])
    chunks = chunker.generate_all_chunks()
    print(chunker.to_json())
    print(f"Generated {len(chunks)} chunks", file=sys.stderr)

if __name__ == "__main__":
    main()