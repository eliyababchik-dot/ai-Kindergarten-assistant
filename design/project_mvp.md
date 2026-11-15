# Staff Scheduling System: MVP

Pain: Kindergarten staff absences and sudden cancellations cause last-minute closures, leaving parents without childcare and forcing them to miss work. Finding replacement personnel relies on slow, tedious manual phone calls.

Product: an assistant for kindergarten coordinators. For each absence, it filters and ranks available assistants using heuristic scoring, conducts automated WhatsApp conversations to confirm availability and extract constraints, and finalizes the placement.

| Feature | Technique | 
| ----- | ----- | 
| Collecting and storing assistant profile data | Pre-prepared files | 
| Reporting sudden absences and opening a request | Ordinary code (not AI) | 
| Initial availability filtering and heuristic ranking | Ordinary code, SQL queries | 
| Managing  dialogue for arrival confirmation | Ordinary code ,LLM call, Multi-turn agent | 
| Re-sorting heuristically and closing the assignment | Ordinary code, SQL UPDATE | 
