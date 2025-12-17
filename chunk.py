"""
AURORA RAG CHUNKER - DETERMINISTIC DATA TRANSFORMATION
Exact schema match + retrieval-aligned anchors
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

    def _event_name(self, event_id):
        row = self.events_master[self.events_master['event_id'] == event_id]
        if row.empty:
            return ""
        return self._get(row.iloc[0], 'event_name')

    # ===================== MAIN =====================

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
        if self.event_details.empty:
            return

        for _, day in self.event_details.iterrows():
            event_id = self._get(day, 'event_id')
            venue_id = self._get(day, 'venue_id')
            date = self._get(day, 'date')
            start_time = self._get(day, 'start_time')
            end_time = self._get(day, 'end_time')

            if not all([event_id, venue_id, date, start_time, end_time]):
                continue

            event_name = self._event_name(event_id)
            venue_row = self.venues[self.venues['venue_id'] == venue_id]
            if venue_row.empty:
                continue

            venue_name = self._get(venue_row.iloc[0], 'venue_name')
            building = self._get(venue_row.iloc[0], 'building')
            date = date.split()[0]

            anchor = f"When is {event_name}?"
            fact = f"{event_name} is scheduled on {date} from {start_time} to {end_time} at {venue_name}, {building}."

            self.chunks.append({
                "text": f"{anchor}\n{fact}",
                "metadata": {
                    "chunk_type": "timing",
                    "event_id": event_id,
                    "venue_id": venue_id,
                    "date": date,
                    "start_time": start_time,
                    "end_time": end_time,
                    "source": ["event_details", "venues", "events_master"],
                    "confidence": 1.0
                }
            })

    # ===================== OVERVIEW =====================

    def _chunk_overview(self):
        for _, event in self.events_master.iterrows():
            event_id = self._get(event, 'event_id')
            event_name = self._get(event, 'event_name')
            if not event_name:
                continue

            anchor = f"Is registration required for {event_name}?"
            fact = (
                f"{event_name} is a {self._get(event,'event_type')} organized by "
                f"{self._get(event,'club_name')}. "
                f"Registration required: {self._get(event,'registration_required')}. "
                f"Certificates offered: {self._get(event,'certificate_offered')}. "
                f"Duration: {self._get(event,'num_days')} day(s)."
            )

            self.chunks.append({
                "text": f"{anchor}\n{fact}",
                "metadata": {
                    "chunk_type": "overview",
                    "event_id": event_id,
                    "source": ["events_master"],
                    "confidence": 1.0
                }
            })

    # ===================== TOPICS / PREREQS =====================

    def _chunk_topics_prerequisites(self):
        for event_id in self.event_details['event_id'].unique():
            row = self.event_details[self.event_details['event_id'] == event_id].iloc[0]
            event_name = self._event_name(event_id)

            topics = self._get(row, 'topics_covered')
            prereq = self._get(row, 'prerequisites')
            pref = self._get(row, 'preferred_knowledge')
            proj = self._get(row, 'project_description')

            if topics and topics.lower() != 'none':
                self.chunks.append({
                    "text": f"What topics are covered in {event_name}?\n{event_name} covers: {topics}.",
                    "metadata": {
                        "chunk_type": "topics",
                        "event_id": event_id,
                        "source": ["event_details"],
                        "confidence": 1.0
                    }
                })

            if prereq and prereq.lower() != 'none':
                txt = f"{event_name} prerequisites: {prereq}."
                if pref and pref.lower() != 'none':
                    txt += f" Preferred knowledge: {pref}."
                self.chunks.append({
                    "text": f"What are the prerequisites for {event_name}?\n{txt}",
                    "metadata": {
                        "chunk_type": "prerequisites",
                        "event_id": event_id,
                        "source": ["event_details"],
                        "confidence": 1.0
                    }
                })

            if proj and proj.lower() != 'none':
                self.chunks.append({
                    "text": f"What project will be built in {event_name}?\n{event_name} project: {proj}.",
                    "metadata": {
                        "chunk_type": "project",
                        "event_id": event_id,
                        "source": ["event_details"],
                        "confidence": 1.0
                    }
                })

    # ===================== FAQ =====================

    def _chunk_faqs(self):
        for _, faq in self.faqs.iterrows():
            q = self._get(faq, 'question')
            a = self._get(faq, 'answer')
            if not q or not a:
                continue

            self.chunks.append({
                "text": f"Q: {q}\nA: {a}",
                "metadata": {
                    "chunk_type": "faq",
                    "faq_id": self._get(faq, 'faq_id'),
                    "event_id": self._get(faq, 'event_id'),
                    "category": self._get(faq, 'category'),
                    "source": ["faqs"],
                    "confidence": 1.0
                }
            })

    # ===================== VENUES =====================

    def _chunk_venues(self):
        for _, v in self.venues.iterrows():
            name = self._get(v, 'venue_name')
            if not name:
                continue

            self.chunks.append({
                "text": f"Where is {name} located?\n{name} is located in {self._get(v,'building')}. Capacity: {self._get(v,'capacity')}. Facilities: {self._get(v,'facilities')}.",
                "metadata": {
                    "chunk_type": "venue",
                    "venue_id": self._get(v, 'venue_id'),
                    "source": ["venues"],
                    "confidence": 1.0
                }
            })

    # ===================== CONTACTS =====================

    def _chunk_contacts(self):
        for _, c in self.contacts.iterrows():
            event_id = self._get(c, 'event_id')
            name = self._get(c, 'name')
            if not name:
                continue

            event_name = "Aurora Fest" if event_id == "ALL" else self._event_name(event_id)

            self.chunks.append({
                "text": f"Who is the contact for {event_name}?\nFor {event_name}, contact {name} ({self._get(c,'role')}). Email: {self._get(c,'email')}. Phone: {self._get(c,'phone')}.",
                "metadata": {
                    "chunk_type": "contact",
                    "contact_id": self._get(c, 'contact_id'),
                    "event_id": event_id,
                    "source": ["contacts"],
                    "confidence": 1.0
                }
            })

    # ===================== CAMPUS =====================

    def _chunk_campus_info(self):
        for _, i in self.campus_info.iterrows():
            title = self._get(i, 'title')
            if not title:
                continue

            self.chunks.append({
                "text": f"Where is the {title}?\n{title}: {self._get(i,'description')}. Location: {self._get(i,'location')}. Hours: {self._get(i,'hours')}.",
                "metadata": {
                    "chunk_type": "campus_info",
                    "info_id": self._get(i, 'info_id'),
                    "category": self._get(i, 'category'),
                    "source": ["campus_info"],
                    "confidence": 1.0
                }
            })

    # ===================== LIVE FLAGS =====================

    def _chunk_live_flags(self):
        for _, f in self.live_flags.iterrows():
            if self._get(f, 'is_active').upper() != 'YES':
                continue

            event_id = self._get(f, 'event_id')
            event_name = "Aurora Fest" if event_id == "ALL" else self._event_name(event_id)

            self.chunks.append({
                "text": f"Is there an update for {event_name}?\nLIVE UPDATE [{self._get(f,'flag_type')}]: {event_name} — {self._get(f,'message')}. Updated at {self._get(f,'created_at')}.",
                "metadata": {
                    "chunk_type": "live_flag",
                    "flag_id": self._get(f, 'flag_id'),
                    "event_id": event_id,
                    "source": ["live_flags"],
                    "confidence": 1.0
                }
            })

    def to_json(self):
        return json.dumps(self.chunks, indent=2, ensure_ascii=False)


def main():
    if len(sys.argv) < 2:
        print("Usage: python chunk.py Aurora_FAQ_KnowledgeBase_v1.xlsx", file=sys.stderr)
        sys.exit(1)

    chunker = AuroraChunker(sys.argv[1])
    chunks = chunker.generate_all_chunks()
    print(chunker.to_json(), end="")
    print(f"\nGenerated {len(chunks)} chunks", file=sys.stderr)


if __name__ == "__main__":
    main()
