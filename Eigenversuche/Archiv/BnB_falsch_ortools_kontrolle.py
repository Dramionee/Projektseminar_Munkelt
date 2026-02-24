from ortools.sat.python import cp_model
import matplotlib.pyplot as plt


jobs = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (2, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(0, 9), (3, 3), (4, 5), (5, 4), (2, 3), (1, 1)],
    [(2, 3), (3, 3), (4, 9), (1, 10), (0, 4), (5, 1)],
]

model = cp_model.CpModel()

# obere schranke
max_zeit = sum(
    dauer
    for job in jobs
    for machine, dauer in job
)

operation_vars = {} # leeres Wörterbuch, speichert jobs, deren reihnfolge und start- und endzeit
machine_intervals = [[] for _ in range(10)] # das gleiche aber für maschienen

for job_idx, job in enumerate(jobs): # schaut sich jeden job einmal an
    for op_idx, (machine, duration) in enumerate(job): # schaut jede operation jedes jobs an, index, maschiene, dauer
        start = model.NewIntVar(0, max_zeit, f'start_j{job_idx}_o{op_idx}') # anlegen startzeit
        end   = model.NewIntVar(0, max_zeit, f'end_j{job_idx}_o{op_idx}') # anlegen endzeit
        operation_vars[job_idx, op_idx] = (start, end) # fügt das in die beiden listen ein
        machine_intervals[machine].append(
            model.NewIntervalVar(start, duration, end, f'interval_j{job_idx}_o{op_idx}')
        )
        if op_idx > 0: # bedingung, dass erst gestartet werden darf, wenn das vorherige fertig ist
            model.Add(start >= operation_vars[job_idx, op_idx-1][1]) # Reihenfolge im Job

for intervals in machine_intervals: # maschienen dürfen nur einen job gleichzeitig bearbeiten
    model.AddNoOverlap(intervals)

makespan = model.NewIntVar(0, max_zeit, 'makespan') # variablendefinition, von 0 bis max zeit
model.AddMaxEquality(makespan, [ # makespan soll gleich dem spätesten Ende aller Jobs sein
    operation_vars[job_idx, len(job)-1][1] for job_idx, job in enumerate(jobs)
])
model.Minimize(makespan) # das maximum, wann alles fertig ist so klein es geht machen

solver = cp_model.CpSolver() # initialisieren
solver.parameters.log_search_progress = True # ganzen dev ausgaben
solver.Solve(model) # lösen

print('Makespan:', solver.Value(makespan)) # ausgabe der makespan
schedule = []
for job_idx, job in enumerate(jobs): # schaut wieder
    for op_idx, (machine, duration) in enumerate(job): # alles an
        start = solver.Value(operation_vars[job_idx, op_idx][0]) # holt startzeit
        print(f'Job{job_idx+1}, Maschine{machine+1}: {start}-{start+duration}') # ausgabe job maschiene start und dauer
        schedule.append([job_idx, machine, start, duration]) # gespeichert für Gantt Graff

fig, ax = plt.subplots(figsize=(8, 5)) # diagramm 8 und 5 ist die größe in zoll
# VARIANTE 1 (Maschinen als y-Achse, Farben pro Job)
'''
y_achse = ['Maschiene 1', 'Maschiene 2', 'Maschiene 3']  # beschriftung y achse
farben = ['tab:blue', 'tab:orange', 'tab:green']
for job_idx, machine, start, duration in schedule:
    ax.broken_barh([(start, duration)], (machine*10, 9), facecolors=farben[job_idx])
    ax.text(start + duration/2, machine*10+5, f'J{job_idx+1}', ha='center', va='center', color='white')
ax.set_yticks([0,10,20])
ax.set_yticklabels(y_achse)
'''

# VARIANTE 2 (Jobs als y-Achse, Farben pro Maschine)
# '''
y_achse = ['Job1', 'Job2', 'Job3'] # beschriftung y achse in variable
farben = ['tab:blue', 'tab:orange', 'tab:green', 'tab:blue', 'tab:orange', 'tab:green', 'tab:green', 'tab:blue', 'tab:orange', 'tab:green'] # erste blau, ...
for job_idx, machine, start, duration in schedule: # geht alle durch
    ax.broken_barh([(start, duration)], ((job_idx+1)*10, 9), facecolors=farben[machine]) # zeichnet horizontalen balken
    ax.text(start + duration/2, (job_idx+1)*10+5, f'M{machine+1}', ha='center', va='center', color='white') # schreibt den text im balken
ax.set_yticks([10,20,30]) # höhe der y achse
ax.set_yticklabels(y_achse) # text auf der y achse (was in variable geschriebe wurde)
# '''

ax.set_xlabel('Zeit') # x achse beschriftung
ax.set_title(f'Makespan={solver.Value(makespan)}') # titel der grafik
plt.tight_layout() # macht alles ordentlich
plt.show()
