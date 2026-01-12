import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Grafici
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Učitavanje CSV-a
student_info = pd.read_csv('studentInfo.csv')
student_vle = pd.read_csv('studentVle.csv')
vle = pd.read_csv('vle.csv')
student_assessment = pd.read_csv('studentAssessment.csv')
assessments = pd.read_csv('assessments.csv')
courses = pd.read_csv('courses.csv')
student_registration = pd.read_csv('studentRegistration.csv')


SELECTED_COURSE = 'BBB'
SELECTED_PRESENTATION = '2013B'  # Prva realizacija

# Filtriranje podataka
df = student_info[
    (student_info['code_module'] == SELECTED_COURSE) &
    (student_info['code_presentation'] == SELECTED_PRESENTATION)
].copy()

print(f"\nFiltered data for {SELECTED_COURSE} {SELECTED_PRESENTATION}:")
print(f"Number of students: {len(df)}")
print("\nFinal result distribution:")
print(df['final_result'].value_counts())
print(f"\nPercentage:")
print(df['final_result'].value_counts(normalize=True) * 100)

def create_target(final_result):
    """
    Pass/Distinction → 0 (negativan - student je uspešan)
    Fail/Withdrawn → 1 (pozitivan - student je u riziku)
    """
    if final_result in ['Pass', 'Distinction']:
        return 0  # Uspešan
    else:
        return 1  # Neuspešan student (target za intervenciju)

# Kreiranje targeta
df['target'] = df['final_result'].apply(create_target)


# Proveri da li imaš dovoljno obe klase
class_balance = df['target'].value_counts(normalize=True)

if class_balance[1] < 0.15 or class_balance[1] > 0.85:
    print("Imbalanced classes! ")

# Demografski podaci iz studentInfo
# Potrebno je enkodirati kategoričke varijable

demographic_features = [
    'gender', 'region', 'highest_education',
    'imd_band', 'age_band', 'num_of_prev_attempts',
    'studied_credits', 'disability'
]

# kopija samo sa demografskim podacima
df_demo = df[demographic_features + ['target', 'id_student']].copy()

# proveri missing values
print("\nMissing values:")
print(df_demo.isnull().sum())

# one-Hot Encoding
df_demo_encoded = pd.get_dummies(
    df_demo,
    columns=[
        'gender', 'region', 'highest_education',
        'imd_band', 'age_band', 'disability'
    ],
    drop_first=True  # da izbegnemo multikolinearnost
)

# izdvajanje x i y
X_demo = df_demo_encoded.drop(['target', 'id_student'], axis=1)
y = df_demo_encoded['target']

print(f"\nX shape: {X_demo.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeature names: {X_demo.columns.tolist()}")

# FAZA 2
print('FAZA 2')

# Korak 2.1: Split podataka na train i test

X_train, X_test, y_train, y_test = train_test_split(
    X_demo, y,
    test_size=0.3,  # 30% za test
    random_state=42,  # Za reproducibilnost
    stratify=y  # Održi balans klasa
)

print("SPLIT PODATAKA")

print(f"Train set: {X_train.shape[0]} studenata")
print(f"Test set: {X_test.shape[0]} studenata")
print(f"\nTrain target distribucija:")
print(y_train.value_counts())
print(f"\nTest target distribucija:")
print(y_test.value_counts())

# 2.2: Decision Tree model

# kreiranje modela model
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  # Ograniči dubinu stabla
    min_samples_split=50  # Minimum studenata za split
)

# treniranje
dt_model.fit(X_train, y_train)

# testiranje
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]

print("REZULTATI - DECISION TREE")


accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_pred_proba_dt)

print(f"\nAccuracy:  {accuracy_dt:.3f} ({accuracy_dt*100:.1f}%)")
print(f"Precision: {precision_dt:.3f}")
print(f"Recall:    {recall_dt:.3f}")
print(f"F1-Score:  {f1_dt:.3f}")
print(f"AUC-ROC:   {auc_dt:.3f}")

print("\nConfusion Matrix:")
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)
print("\nInterpretacija:")
print(f"  True Negatives (TN):  {cm_dt[0,0]}")
print(f"  False Positives (FP): {cm_dt[0,1]}")
print(f"  False Negatives (FN): {cm_dt[1,0]}")
print(f"  True Positives (TP):  {cm_dt[1,1]}")

# Feature importance
print("\nNajvažniji faktori (Top 10):")
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:40s}: {row['Importance']:.4f}")

# 2.3: Logistic Regression model


print("TRENIRANJE: LOGISTIC REGRESSION")

# Kreiraj model
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

# TRENIRANJE
lr_model.fit(X_train, y_train)

# TESTIRANJE
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"\nAccuracy:  {accuracy_lr:.3f} ({accuracy_lr*100:.1f}%)")
print(f"Precision: {precision_lr:.3f}")
print(f"Recall:    {recall_lr:.3f}")
print(f"F1-Score:  {f1_lr:.3f}")
print(f"AUC-ROC:   {auc_lr:.3f}")

print("\nConfusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(cm_lr)
print("\nInterpretacija:")
print(f"  True Negatives (TN):  {cm_lr[0,0]}")
print(f"  False Positives (FP): {cm_lr[0,1]}")
print(f"  False Negatives (FN): {cm_lr[1,0]}")
print(f"  True Positives (TP):  {cm_lr[1,1]}")


# POREĐENJE MODELA
comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'Logistic Regression'],
    'Accuracy': [accuracy_dt, accuracy_lr],
    'Precision': [precision_dt, precision_lr],
    'Recall': [recall_dt, recall_lr],
    'F1-Score': [f1_dt, f1_lr],
    'AUC': [auc_dt, auc_lr]
})

print("\n", comparison.to_string(index=False))
print(f"• Bolji Recall: {'Decision Tree' if recall_dt > recall_lr else 'Logistic Regression'}")
print(f"• Bolji Precision: {'Decision Tree' if precision_dt > precision_lr else 'Logistic Regression'}")
print(f"• Bolji AUC: {'Decision Tree' if auc_dt > auc_lr else 'Logistic Regression'}")


# VIZUALIZACIJE

# 1. ROC Curve

plt.figure(figsize=(10, 6))

# Decision Tree ROC
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc_dt:.3f})', linewidth=2)

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.3f})', linewidth=2)

# Baseline (random)
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', linewidth=1)

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Demographics Only', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 2. Confusion Matrices

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision Tree
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Success', 'At Risk'],
            yticklabels=['Success', 'At Risk'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_title('Decision Tree\nConfusion Matrix', fontsize=13, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

# Logistic Regression
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Success', 'At Risk'],
            yticklabels=['Success', 'At Risk'],
            ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_title('Logistic Regression\nConfusion Matrix', fontsize=13, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.show()


# 3. Metrics Comparison Bar Chart

fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
dt_scores = [accuracy_dt, precision_dt, recall_dt, f1_dt, auc_dt]
lr_scores = [accuracy_lr, precision_lr, recall_lr, f1_lr, auc_lr]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, dt_scores, width, label='Decision Tree', color='steelblue')
bars2 = ax.bar(x + width/2, lr_scores, width, label='Logistic Regression', color='seagreen')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison - Demographics Only', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=11)
ax.set_ylim(0, 1)
ax.grid(True, axis='y', alpha=0.3)

# dodavanje vrednosti na stubićima
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# 4. Feature Importance Comparison

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Decision Tree - Top 10
top_features_dt = feature_importance.head(10)
axes[0].barh(range(len(top_features_dt)), top_features_dt['Importance'], color='steelblue')
axes[0].set_yticks(range(len(top_features_dt)))
axes[0].set_yticklabels(top_features_dt['Feature'], fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Importance', fontsize=11)
axes[0].set_title('Decision Tree\nTop 10 Most Important Features', fontsize=12, fontweight='bold')
axes[0].grid(True, axis='x', alpha=0.3)

# Logistic Regression - Top 10 (by absolute coefficient)
feature_coef = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_[0]
})
feature_coef['Abs_Coef'] = feature_coef['Coefficient'].abs()
top_features_lr = feature_coef.nlargest(10, 'Abs_Coef')

colors = ['crimson' if c > 0 else 'seagreen' for c in top_features_lr['Coefficient']]
axes[1].barh(range(len(top_features_lr)), top_features_lr['Coefficient'], color=colors)
axes[1].set_yticks(range(len(top_features_lr)))
axes[1].set_yticklabels(top_features_lr['Feature'], fontsize=9)
axes[1].invert_yaxis()
axes[1].set_xlabel('Coefficient (Red=Risk↑, Green=Risk↓)', fontsize=11)
axes[1].set_title('Logistic Regression\nTop 10 Most Influential Features', fontsize=12, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.show()


# FAZA 3 sa ponasanjem

print("FAZA 3")

#3.1: ucitavanje i filtriranje VLE podataka

vle_filtered = student_vle[
    (student_vle['code_module'] == SELECTED_COURSE) &
    (student_vle['code_presentation'] == SELECTED_PRESENTATION)
].copy()

print(f"✓ VLE podaci učitani: {vle_filtered.shape[0]:,} zapisa")
print(f"  Broj jedinstvenih studenata: {vle_filtered['id_student'].nunique()}")
print(f"  Datum opseg: {vle_filtered['date'].min()} do {vle_filtered['date'].max()} dana")
print(f"  Ukupno klikova: {vle_filtered['sum_click'].sum():,}")

#koliko studenata nema VLE aktivnost
students_with_vle = set(vle_filtered['id_student'].unique())
all_students = set(df['id_student'].unique())
students_without_vle = all_students - students_with_vle

print(f"Studenti BEZ VLE aktivnosti: {len(students_without_vle)} ({len(students_without_vle)/len(all_students)*100:.1f}%)")

# 3.2: Izbor vremenskog prozora

TIME_WINDOW = 15

print(f"   Izabran prozor: PRVIH {TIME_WINDOW} DANA kursa")

# Filtriranje samo prvih dana
vle_early = vle_filtered[vle_filtered['date'] <= TIME_WINDOW].copy()

print(f"\n✓ VLE podaci za prvih {TIME_WINDOW} dana:")
print(f"  Zapisi: {vle_early.shape[0]:,}")
print(f"  Studenti sa aktivnošću: {vle_early['id_student'].nunique()}")
print(f"  Ukupno klikova: {vle_early['sum_click'].sum():,}")
print(f"  Prosečno klikova po studentu: {vle_early.groupby('id_student')['sum_click'].sum().mean():.1f}")

# 3.3: Spajanje sa activity_type iz vle tabele

vle_with_activity = vle_early.merge(
    vle[['id_site', 'activity_type']],
    on='id_site',
    how='left'
)

print(vle_with_activity['activity_type'].value_counts().head(10))

# 3.4: Feature Engineering - Agregacije

# FEATURE 1: Total clicks per student
print("  [1/8] Total clicks...")
total_clicks = vle_early.groupby('id_student').agg({
    'sum_click': 'sum'
}).reset_index()
total_clicks.columns = ['id_student', 'total_clicks']

# FEATURE 2: Number of distinct days active
print("  [2/8] Days active...")
days_active = vle_early.groupby('id_student')['date'].nunique().reset_index()
days_active.columns = ['id_student', 'days_active']

# FEATURE 3: Average clicks per day
print("  [3/8] Average clicks per day...")
avg_clicks_per_day = vle_early.groupby('id_student').agg({
    'sum_click': 'mean'
}).reset_index()
avg_clicks_per_day.columns = ['id_student', 'avg_clicks_per_day']

# FEATURE 4: Number of different materials accessed
print("  [4/8] Number of materials accessed...")
materials_accessed = vle_early.groupby('id_student')['id_site'].nunique().reset_index()
materials_accessed.columns = ['id_student', 'num_materials']

# FEATURE 5: Clicks by activity type (PIVOT)
print("  [5/8] Clicks by activity type...")
clicks_by_activity = vle_with_activity.pivot_table(
    index='id_student',
    columns='activity_type',
    values='sum_click',
    aggfunc='sum',
    fill_value=0
).reset_index()

# Rename columns
clicks_by_activity.columns = ['id_student'] + [f'clicks_{col}' for col in clicks_by_activity.columns[1:]]

# FEATURE 6: Early engagement (first 3 days)
print("  [6/8] Early engagement (first 3 days)...")
early_engagement = vle_early[vle_early['date'] <= 3].groupby('id_student').agg({
    'sum_click': 'sum'
}).reset_index()
early_engagement.columns = ['id_student', 'clicks_first_3_days']

# FEATURE 7: Late engagement (after day 3)
print("  [7/8] Late engagement (after day 3)...")
late_engagement = vle_early[vle_early['date'] > 3].groupby('id_student').agg({
    'sum_click': 'sum'
}).reset_index()
late_engagement.columns = ['id_student', 'clicks_after_3_days']

# FEATURE 8: Engagement trend (late/early ratio)
print("  [8/8] Engagement trend...")

#3.5: Spajanje svih feature-a

vle_features = total_clicks
vle_features = vle_features.merge(days_active, on='id_student', how='left')
vle_features = vle_features.merge(avg_clicks_per_day, on='id_student', how='left')
vle_features = vle_features.merge(materials_accessed, on='id_student', how='left')
vle_features = vle_features.merge(clicks_by_activity, on='id_student', how='left')
vle_features = vle_features.merge(early_engagement, on='id_student', how='left')
vle_features = vle_features.merge(late_engagement, on='id_student', how='left')

# Fill NaN sa 0 (studenti bez aktivnosti)
vle_features = vle_features.fillna(0)

# engagement trend
vle_features['engagement_trend'] = np.where(
    vle_features['clicks_first_3_days'] > 0,
    vle_features['clicks_after_3_days'] / vle_features['clicks_first_3_days'],
    0
)

print(f"✓ VLE feature-i kreirani: {vle_features.shape}")
print(f"  Broj studenata: {len(vle_features)}")
print(f"  Broj novih feature-a: {vle_features.shape[1] - 1}")  # -1 za id_student

print("\nNovi VLE feature-i:")
for col in vle_features.columns:
    if col != 'id_student':
        print(f"  • {col}")

# 3.6: Spajanje sa demografskim podacima

# Demographics + VLE features
df_full = df[demographic_features + ['target', 'id_student']].merge(
    vle_features,
    on='id_student',
    how='left'  # LEFT JOIN - zadržavamo SVE studente
)

# Fill NaN (studenti bez VLE aktivnosti) sa 0
vle_feature_cols = [col for col in vle_features.columns if col != 'id_student']
df_full[vle_feature_cols] = df_full[vle_feature_cols].fillna(0)

print(f"  Ukupno studenata: {len(df_full)}")
print(f"  Ukupno kolona: {df_full.shape[1]}")
print(f"  Missing values: {df_full.isnull().sum().sum()}")

df_full_encoded = pd.get_dummies(
    df_full,
    columns=[
        'gender', 'region', 'highest_education',
        'imd_band', 'age_band', 'disability'
    ],
    drop_first=True
)

# X i y
X_full = df_full_encoded.drop(['target', 'id_student'], axis=1)
y_full = df_full_encoded['target']

print(f"  X_full shape: {X_full.shape}")
print(f"  Broj feature-a: {X_full.shape[1]}")
print(f"    └─ Demographics: ~30")
print(f"    └─ VLE behavior: ~{X_full.shape[1] - 30}")




# FAZA 4

print('FAZA 4')

# 4.1: Split podataka

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full,
    test_size=0.3,
    random_state=42,
    stratify=y_full
)

print(f"\nTrain set: {X_train_full.shape[0]} studenata")
print(f"Test set: {X_test_full.shape[0]} studenata")
print(f"Broj feature-a: {X_train_full.shape[1]}")

# 4.2: Decision Tree sa VLE podacima

dt_model_full = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_split=50
)

#treniranje
dt_model_full.fit(X_train_full, y_train_full)

#testiranje
y_pred_dt_full = dt_model_full.predict(X_test_full)
y_pred_proba_dt_full = dt_model_full.predict_proba(X_test_full)[:, 1]

accuracy_dt_full = accuracy_score(y_test_full, y_pred_dt_full)
precision_dt_full = precision_score(y_test_full, y_pred_dt_full)
recall_dt_full = recall_score(y_test_full, y_pred_dt_full)
f1_dt_full = f1_score(y_test_full, y_pred_dt_full)
auc_dt_full = roc_auc_score(y_test_full, y_pred_proba_dt_full)

print(f"\nAccuracy:  {accuracy_dt_full:.3f} ({accuracy_dt_full*100:.1f}%)")
print(f"Precision: {precision_dt_full:.3f}")
print(f"Recall:    {recall_dt_full:.3f}")
print(f"F1-Score:  {f1_dt_full:.3f}")
print(f"AUC-ROC:   {auc_dt_full:.3f}")

print("\nConfusion Matrix:")
cm_dt_full = confusion_matrix(y_test_full, y_pred_dt_full)
print(cm_dt_full)
print("\nInterpretacija:")
print(f"  True Negatives (TN):  {cm_dt_full[0,0]}")
print(f"  False Positives (FP): {cm_dt_full[0,1]}")
print(f"  False Negatives (FN): {cm_dt_full[1,0]}")
print(f"  True Positives (TP):  {cm_dt_full[1,1]}")

# Feature importance
print("\nNajvažniji faktori (Top 15):")
feature_importance_full = pd.DataFrame({
    'Feature': X_train_full.columns,
    'Importance': dt_model_full.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in feature_importance_full.head(15).iterrows():
    print(f"  {row['Feature']:40s}: {row['Importance']:.4f}")

# 4.3: Logistic Regression sa VLE podacima

lr_model_full = LogisticRegression(
    random_state=42,
    max_iter=1000
)

# treniranje
lr_model_full.fit(X_train_full, y_train_full)

# testiranje
y_pred_lr_full = lr_model_full.predict(X_test_full)
y_pred_proba_lr_full = lr_model_full.predict_proba(X_test_full)[:, 1]


accuracy_lr_full = accuracy_score(y_test_full, y_pred_lr_full)
precision_lr_full = precision_score(y_test_full, y_pred_lr_full)
recall_lr_full = recall_score(y_test_full, y_pred_lr_full)
f1_lr_full = f1_score(y_test_full, y_pred_lr_full)
auc_lr_full = roc_auc_score(y_test_full, y_pred_proba_lr_full)

print(f"\nAccuracy:  {accuracy_lr_full:.3f} ({accuracy_lr_full*100:.1f}%)")
print(f"Precision: {precision_lr_full:.3f}")
print(f"Recall:    {recall_lr_full:.3f}")
print(f"F1-Score:  {f1_lr_full:.3f}")
print(f"AUC-ROC:   {auc_lr_full:.3f}")

print("\nConfusion Matrix:")
cm_lr_full = confusion_matrix(y_test_full, y_pred_lr_full)
print(cm_lr_full)
print("\nInterpretacija:")
print(f"  True Negatives (TN):  {cm_lr_full[0,0]}")
print(f"  False Positives (FP): {cm_lr_full[0,1]}")
print(f"  False Negatives (FN): {cm_lr_full[1,0]} ")
print(f"  True Positives (TP):  {cm_lr_full[1,1]}")

# Poredjenje: Demographics vs Demographics+VLE

print("\n" + "=" * 60)
print("POREĐENJE: DEMOGRAPHICS vs DEMOGRAPHICS + VLE")
print("=" * 60)

comparison_full = pd.DataFrame({
    'Model': [
        'Decision Tree (Demo only)',
        'Decision Tree (Demo + VLE)',
        'Logistic Reg (Demo only)',
        'Logistic Reg (Demo + VLE)'
    ],
    'Recall': [recall_dt, recall_dt_full, recall_lr, recall_lr_full],
    'Precision': [precision_dt, precision_dt_full, precision_lr, precision_lr_full],
    'F1-Score': [f1_dt, f1_dt_full, f1_lr, f1_lr_full],
    'AUC': [auc_dt, auc_dt_full, auc_lr, auc_lr_full],
    'FN (Propušteni)': [
        cm_dt[1,0], cm_dt_full[1,0],
        cm_lr[1,0], cm_lr_full[1,0]
    ]
})

print("\n", comparison_full.to_string(index=False))

print(f"\n1. Logistic Regression poboljšanje:")
print(f"   Recall: {recall_lr:.3f} → {recall_lr_full:.3f} (Δ = {recall_lr_full - recall_lr:+.3f})")
print(f"   False Negatives: {cm_lr[1,0]} → {cm_lr_full[1,0]} (Δ = {cm_lr_full[1,0] - cm_lr[1,0]:+d} studenata)")

print(f"\n2. Decision Tree poboljšanje:")
print(f"   Recall: {recall_dt:.3f} → {recall_dt_full:.3f} (Δ = {recall_dt_full - recall_dt:+.3f})")
print(f"   False Negatives: {cm_dt[1,0]} → {cm_dt_full[1,0]} (Δ = {cm_dt_full[1,0] - cm_dt[1,0]:+d} studenata)")

# FEATURE IMPORTANCE - DEMOGRAPHICS + VLE

print("FEATURE IMPORTANCE - SA VLE PODACIMA")

# Decision Tree - Top 15
print("\nDecision Tree (Demographics + VLE) - Top 15:")
feature_importance_dt = pd.DataFrame({
    'Feature': X_train_full.columns,
    'Importance': dt_model_full.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in feature_importance_dt.head(15).iterrows():
    print(f"  {row['Feature']:40s}: {row['Importance']:.4f}")

# Logistic Regression - Top 15
print("\nLogistic Regression (Demographics + VLE) - Top 15:")
feature_coef_lr = pd.DataFrame({
    'Feature': X_train_full.columns,
    'Coefficient': lr_model_full.coef_[0],
    'Abs_Coef': np.abs(lr_model_full.coef_[0])
}).sort_values('Abs_Coef', ascending=False)

for i, row in feature_coef_lr.head(15).iterrows():
    direction = "↑ Risk" if row['Coefficient'] > 0 else "↓ Risk"
    print(f"  {row['Feature']:40s}: {row['Coefficient']:+.4f} ({direction})")

# Vizualizacija
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Decision Tree
top_dt = feature_importance_dt.head(15)
axes[0].barh(range(len(top_dt)), top_dt['Importance'], color='steelblue')
axes[0].set_yticks(range(len(top_dt)))
axes[0].set_yticklabels(top_dt['Feature'], fontsize=9)
axes[0].invert_yaxis()
axes[0].set_xlabel('Importance', fontsize=11)
axes[0].set_title('Decision Tree - Top 15 Features\n(Demographics + VLE)',
                  fontsize=12, fontweight='bold')
axes[0].grid(True, axis='x', alpha=0.3)

# Logistic Regression
top_lr = feature_coef_lr.head(15)
colors = ['crimson' if c > 0 else 'seagreen' for c in top_lr['Coefficient']]
axes[1].barh(range(len(top_lr)), top_lr['Coefficient'], color=colors)
axes[1].set_yticks(range(len(top_lr)))
axes[1].set_yticklabels(top_lr['Feature'], fontsize=9)
axes[1].invert_yaxis()
axes[1].set_xlabel('Coefficient (Red=Risk↑, Green=Risk↓)', fontsize=11)
axes[1].set_title('Logistic Regression - Top 15 Features\n(Demographics + VLE)',
                  fontsize=12, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# FAZA 5: THRESHOLD OPTIMIZATION

print("FAZA 5")

print(" Default threshold = 0.5 (50% verovatnoća)")

# 5.1: Analiza različitih threshold-a

# Koristićemo Logistic Regression
# y_pred_proba_lr_full već imamo (verovatnoće)

thresholds_to_test = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

results_threshold = []

for threshold in thresholds_to_test:
    # Predict sa custom threshold-om
    y_pred_custom = (y_pred_proba_lr_full >= threshold).astype(int)

    # Metrics
    acc = accuracy_score(y_test_full, y_pred_custom)
    prec = precision_score(y_test_full, y_pred_custom)
    rec = recall_score(y_test_full, y_pred_custom)
    f1 = f1_score(y_test_full, y_pred_custom)

    # Confusion matrix
    cm = confusion_matrix(y_test_full, y_pred_custom)
    tn, fp, fn, tp = cm.ravel()

    results_threshold.append({
        'Threshold': threshold,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    })

# DataFrame sa rezultatima
df_threshold = pd.DataFrame(results_threshold)

print("\n Rezultati za različite threshold-ove:\n")
print(df_threshold.to_string(index=False))

# 5.2: Vizualizacija Precision-Recall trade-off

print("Precision vs Recall Trade-off")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Recall vs Threshold
axes[0, 0].plot(df_threshold['Threshold'], df_threshold['Recall'],
                marker='o', linewidth=2, markersize=8, color='green')
axes[0, 0].set_xlabel('Threshold', fontsize=11)
axes[0, 0].set_ylabel('Recall', fontsize=11)
axes[0, 0].set_title('Recall vs Threshold', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0.80, color='red', linestyle='--', label='Target 80%')
axes[0, 0].legend()

# Plot 2: Precision vs Threshold
axes[0, 1].plot(df_threshold['Threshold'], df_threshold['Precision'],
                marker='o', linewidth=2, markersize=8, color='blue')
axes[0, 1].set_xlabel('Threshold', fontsize=11)
axes[0, 1].set_ylabel('Precision', fontsize=11)
axes[0, 1].set_title('Precision vs Threshold', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: False Negatives vs Threshold
axes[1, 0].plot(df_threshold['Threshold'], df_threshold['FN'],
                marker='o', linewidth=2, markersize=8, color='red')
axes[1, 0].set_xlabel('Threshold', fontsize=11)
axes[1, 0].set_ylabel('False Negatives (Propušteni)', fontsize=11)
axes[1, 0].set_title('False Negatives vs Threshold', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: F1-Score vs Threshold
axes[1, 1].plot(df_threshold['Threshold'], df_threshold['F1-Score'],
                marker='o', linewidth=2, markersize=8, color='purple')
axes[1, 1].set_xlabel('Threshold', fontsize=11)
axes[1, 1].set_ylabel('F1-Score', fontsize=11)
axes[1, 1].set_title('F1-Score vs Threshold', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5.3: Izbor optimalnog threshold-a

print("\n" + "=" * 60)
print("IZBOR OPTIMALNOG THRESHOLD-A")

# traziti threshold sa Recall >= 0.80
optimal_candidates = df_threshold[df_threshold['Recall'] >= 0.80]

if len(optimal_candidates) > 0:
    # od kandidata, izabrati onaj sa najboljim F1-Score-om
    optimal_row = optimal_candidates.loc[optimal_candidates['F1-Score'].idxmax()]
    optimal_threshold = optimal_row['Threshold']

    print(f"\n✓ Optimalan threshold: {optimal_threshold}")
    print(f"\nPerformanse sa threshold = {optimal_threshold}:")
    print(f"   Recall:    {optimal_row['Recall']:.3f} ({optimal_row['Recall'] * 100:.1f}%)")
    print(f"   Precision: {optimal_row['Precision']:.3f} ({optimal_row['Precision'] * 100:.1f}%)")
    print(f"   F1-Score:  {optimal_row['F1-Score']:.3f}")
    print(f"   Accuracy:  {optimal_row['Accuracy']:.3f}")

    print(f"\n   True Positives (TP):   {optimal_row['TP']:.0f} ")
    print(f"   False Negatives (FN):  {optimal_row['FN']:.0f} ")
    print(f"   False Positives (FP):  {optimal_row['FP']:.0f}")
    print(f"   True Negatives (TN):   {optimal_row['TN']:.0f}")

else:
    print("\n  Nijedan threshold ne postiže Recall >= 80%")

    optimal_row = df_threshold.loc[df_threshold['Recall'].idxmax()]
    optimal_threshold = optimal_row['Threshold']

    print(f"\n   Najbolji threshold: {optimal_threshold}")
    print(f"   Recall: {optimal_row['Recall']:.3f} ({optimal_row['Recall'] * 100:.1f}%)")

# 5.4: Poređenje default vs optimal

print("DEFAULT (0.5) vs OPTIMAL THRESHOLD")

default_row = df_threshold[df_threshold['Threshold'] == 0.5].iloc[0]

comparison_threshold = pd.DataFrame({
    'Scenario': ['Default (0.5)', f'Optimal ({optimal_threshold})'],
    'Recall': [default_row['Recall'], optimal_row['Recall']],
    'Precision': [default_row['Precision'], optimal_row['Precision']],
    'F1-Score': [default_row['F1-Score'], optimal_row['F1-Score']],
    'FN (Propušteni)': [default_row['FN'], optimal_row['FN']],
    'FP (False Alarms)': [default_row['FP'], optimal_row['FP']]
})

print("\n", comparison_threshold.to_string(index=False))

print("\n POBOLJŠANJE:")
print(f"   Recall: {default_row['Recall']:.3f} → {optimal_row['Recall']:.3f} "
      f"(Δ = {optimal_row['Recall'] - default_row['Recall']:+.3f})")
print(f"   FN: {default_row['FN']:.0f} → {optimal_row['FN']:.0f} "
      f"(Δ = {optimal_row['FN'] - default_row['FN']:+.0f} studenata)")
print(f"   FP: {default_row['FP']:.0f} → {optimal_row['FP']:.0f} "
      f"(Δ = {optimal_row['FP'] - default_row['FP']:+.0f} studenata)")

# Skaliranje na 1000 studenata
scale_factor = 1000 / len(y_test_full)

fn_saved = (default_row['FN'] - optimal_row['FN']) * scale_factor
fp_added = (optimal_row['FP'] - default_row['FP']) * scale_factor

print(f"\n Na 1000 studenata:")
print(f"   Spašeno dodatnih {fn_saved:.0f} studenata u riziku!")


# FAZA 6

print("FAZA 6: KLASTEROVANJE - GRUPISANJE STUDENATA")

# 6.1: Priprema podataka za klasterovanje

# demographics + VLE features (X_full)
# samo studente sa VLE aktivnostima

# Identifikovanje studenata sa VLE
students_with_activity = df_full[df_full['total_clicks'] > 0]['id_student'].values

# Filtriranje X_full i y_full
X_cluster = X_full[df_full_encoded['id_student'].isin(students_with_activity)].copy()
y_cluster = y_full[df_full_encoded['id_student'].isin(students_with_activity)].copy()
ids_cluster = df_full_encoded[df_full_encoded['id_student'].isin(students_with_activity)]['id_student'].values

print(f"\n✓ Podaci za klasterovanje:")
print(f"  Broj studenata: {len(X_cluster)}")
print(f"  Broj feature-a: {X_cluster.shape[1]}")
print(f"  Studenti sa VLE aktivnošću: {len(students_with_activity)}")

# Izdvoji samo VLE feature-e za klasterovanje
# (demografija je manje važna za grupisanje ponašanja)
vle_feature_names = [col for col in X_full.columns if
                     'clicks' in col or 'days_active' in col or 'num_materials' in col or 'engagement_trend' in col]

X_cluster_vle = X_cluster[vle_feature_names].copy()

print(f"\n  VLE feature-i za klasterovanje: {len(vle_feature_names)}")
for feat in vle_feature_names:
    print(f"    • {feat}")

# 6.2: Standardizacija (za K-Means)

print("STANDARDIZACIJA")

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster_vle)

print(
    f"  BEFORE: total_clicks prosek = {X_cluster_vle['total_clicks'].mean():.1f}, std = {X_cluster_vle['total_clicks'].std():.1f}")
print(f"  AFTER:  total_clicks prosek = {X_cluster_scaled[:, 0].mean():.1f}, std = {X_cluster_scaled[:, 0].std():.1f}")

# 6.3: Određivanje optimalnog broja klastera

print("ODREĐIVANJE OPTIMALNOG BROJA KLASTERA")

print("1. Elbow Method - traži 'lakat' na grafikonu")
print("2. Silhouette Score - meri koliko su klasteri dobro odvojeni")

# Testiranje različitih brojeva klastera
k_range = range(2, 11)
inertias = []
silhouette_scores = []

print("\n Testiranje K (2-10)")

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))

    print(f"   K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouette_scores[-1]:.3f}")

# Vizualizacija
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
axes[0].set_xlabel('Broj klastera (K)', fontsize=11)
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=11)
axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Silhouette plot
axes[1].plot(k_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('Broj klastera (K)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Score', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Automatski izbor - max silhouette score
optimal_k = k_range[np.argmax(silhouette_scores)]
optimal_silhouette = max(silhouette_scores)

print(f"\n✓ Optimalan broj klastera: K = {optimal_k}")
print(f"  Silhouette Score: {optimal_silhouette:.3f}")

# 6.4: Finalno klasterovanje

print(f"FINALNO KLASTEROVANJE - K = {optimal_k}")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_cluster_scaled)

X_cluster_vle['Cluster'] = cluster_labels
X_cluster_vle['target'] = y_cluster.values
X_cluster_vle['id_student'] = ids_cluster

print(f"\n✓ Klasterovanje završeno!")
print(f"  Broj klastera: {optimal_k}")
print(f"\n  Raspodela studenata po klasterima:")
for i in range(optimal_k):
    count = (cluster_labels == i).sum()
    print(f"    Cluster {i}: {count} studenata ({count / len(cluster_labels) * 100:.1f}%)")


# 6.5: Analiza klastera (Centroidi)


print("ANALIZA KLASTERA - CENTROIDI")

# Centroidi u originalnim jedinicama (inverse transform)
centroids_scaled = kmeans_final.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

# DataFrame sa centroidima
centroids_df = pd.DataFrame(
    centroids_original,
    columns=vle_feature_names
)
centroids_df['Cluster'] = range(optimal_k)

print("\n Centroidi (prosečne vrednosti po klasteru):\n")

for i in range(optimal_k):
    print(f"\n{'=' * 50}")
    print(f"CLUSTER {i}")
    print(f"{'=' * 50}")

    cluster_data = X_cluster_vle[X_cluster_vle['Cluster'] == i]
    cluster_size = len(cluster_data)
    risk_rate = (cluster_data['target'] == 1).mean()

    print(f"Broj studenata: {cluster_size}")
    print(f"Stopa rizika: {risk_rate * 100:.1f}%")
    print(f"\nProsečne vrednosti:")
    print(f"  Total clicks: {centroids_df.loc[i, 'total_clicks']:.0f}")
    print(f"  Days active: {centroids_df.loc[i, 'days_active']:.1f}")
    print(f"  Avg clicks/day: {centroids_df.loc[i, 'avg_clicks_per_day']:.1f}")
    print(f"  Materials accessed: {centroids_df.loc[i, 'num_materials']:.0f}")
    print(f"  Clicks (first 3 days): {centroids_df.loc[i, 'clicks_first_3_days']:.0f}")
    print(f"  Engagement trend: {centroids_df.loc[i, 'engagement_trend']:.2f}")

# 6.6: Imenovanje klastera

print("INTERPRETACIJA - IMENOVANJE KLASTERA")

# Automatsko imenovanje na osnovu karakteristika
cluster_names = {}
cluster_descriptions = {}

for i in range(optimal_k):
    cluster_data = X_cluster_vle[X_cluster_vle['Cluster'] == i]

    # Karakteristike klastera
    total_clicks = centroids_df.loc[i, 'total_clicks']
    days_active = centroids_df.loc[i, 'days_active']
    engagement_trend = centroids_df.loc[i, 'engagement_trend']
    risk_rate = (cluster_data['target'] == 1).mean()
    avg_clicks_per_day = centroids_df.loc[i, 'avg_clicks_per_day']

    if risk_rate >= 0.45:  # Visok rizik (≥45%)
        if total_clicks < 150:
            name = "Nisko angažovani (Visok rizik)"
        else:
            name = "Umereno angažovani (Srednji rizik)"

    elif risk_rate >= 0.30:  # Srednji rizik (30-45%)
        if engagement_trend < 1.0:
            name = "Opada aktivnost (Srednji rizik)"
        else:
            name = "Stabilno angažovani (Nizak rizik)"
    else:  # Nizak rizik (<30%)
        if total_clicks >= 300:
            name = "Visoko angažovani (Nizak rizik)"
        else:
            name = "Stabilno angažovani (Nizak rizik)"

    cluster_names[i] = name


    print(f"\n{'=' * 50}")
    print(f"CLUSTER {i}: {name}")
    print(f"{'=' * 50}")
    print(f"Stopa rizika: {risk_rate * 100:.1f}%")
    print(f"\nKarakteristike:")
    print(f"  • Total clicks: {total_clicks:.0f}")
    print(f"  • Days active: {days_active:.1f}")
    print(f"  • Avg clicks/day: {avg_clicks_per_day:.1f}")
    print(f"  • Engagement trend: {engagement_trend:.2f}")


#VIZUALIZACIJA KLASTERA

# 1. Scatter plot: Total Clicks vs Days Active

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# plot 1: klasteri po boji
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for i in range(optimal_k):
    cluster_data = X_cluster_vle[X_cluster_vle['Cluster'] == i]

    axes[0].scatter(
        cluster_data['total_clicks'],
        cluster_data['days_active'],
        c=colors[i % len(colors)],
        label=cluster_names[i],
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )

# centroidi
for i in range(optimal_k):
    axes[0].scatter(
        centroids_df.loc[i, 'total_clicks'],
        centroids_df.loc[i, 'days_active'],
        c='black',
        marker='X',
        s=300,
        edgecolors='white',
        linewidth=2,
        label=f'Centroid {i}' if i == 0 else ''
    )

axes[0].set_xlabel('Total Clicks (prvih 15 dana)', fontsize=12)
axes[0].set_ylabel('Days Active (prvih 15 dana)', fontsize=12)
axes[0].set_title('Klasteri studenata po aktivnosti', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9, loc='best')
axes[0].grid(True, alpha=0.3)

# Plot 2: risk rate po klasteru
axes[1].scatter(
    X_cluster_vle['total_clicks'],
    X_cluster_vle['days_active'],
    c=X_cluster_vle['target'],
    cmap='RdYlGn_r',  # Red=Rizik, Green=Uspeh
    alpha=0.6,
    s=50,
    edgecolors='black',
    linewidth=0.5
)

axes[1].set_xlabel('Total Clicks', fontsize=12)
axes[1].set_ylabel('Days Active', fontsize=12)
axes[1].set_title('Studenti po riziku (Crveno=Rizik, Zeleno=Uspeh)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('Target (0=Uspeh, 1=Rizik)', fontsize=10)

plt.tight_layout()
plt.show()

# 2. Box plot: Distribucija po klasterima

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

X_cluster_vle['Cluster_Name'] = X_cluster_vle['Cluster'].map(cluster_names)

# Plot 1: Total Clicks
sns.boxplot(data=X_cluster_vle, x='Cluster_Name', y='total_clicks', ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Total Clicks po klasteru', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Total Clicks', fontsize=10)
axes[0, 0].tick_params(axis='x', rotation=15)

# Plot 2: Days Active
sns.boxplot(data=X_cluster_vle, x='Cluster_Name', y='days_active', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Days Active po klasteru', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('Days Active', fontsize=10)
axes[0, 1].tick_params(axis='x', rotation=15)

# Plot 3: Avg Clicks per Day
sns.boxplot(data=X_cluster_vle, x='Cluster_Name', y='avg_clicks_per_day', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Avg Clicks per Day po klasteru', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('')
axes[1, 0].set_ylabel('Avg Clicks/Day', fontsize=10)
axes[1, 0].tick_params(axis='x', rotation=15)

# Plot 4: Engagement Trend
sns.boxplot(data=X_cluster_vle, x='Cluster_Name', y='engagement_trend', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Engagement Trend po klasteru', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('Engagement Trend', fontsize=10)
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

# 3. Bar chart: Risk rate po klasteru

fig, ax = plt.subplots(figsize=(10, 6))

# risk rate po klasteru
risk_rates = []
cluster_labels = []

for i in range(optimal_k):
    cluster_data = X_cluster_vle[X_cluster_vle['Cluster'] == i]
    risk_rate = (cluster_data['target'] == 1).mean()
    risk_rates.append(risk_rate)
    cluster_labels.append(f"{cluster_names[i]}\n({len(cluster_data)} studenata)")

# bar chart
bars = ax.bar(range(optimal_k), risk_rates, color=colors[:optimal_k], edgecolor='black', linewidth=1.5)

# dodavanje vrednosti na stubićima
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{risk_rates[i] * 100:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Stopa rizika (% studenata u riziku)', fontsize=12)
ax.set_title('Stopa rizika po klasteru', fontsize=14, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels(cluster_labels, fontsize=10)
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='50% threshold')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
