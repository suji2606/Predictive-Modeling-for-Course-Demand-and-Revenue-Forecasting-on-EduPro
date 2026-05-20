import pandas as pd
import numpy as np

def load_and_merge(filepath="EduPro_Online_Platform.xlsx"):
    xl = pd.read_excel(filepath, sheet_name=None)
    courses = xl["Courses"]
    teachers = xl["Teachers"]
    transactions = xl["Transactions"]

    # Aggregate transactions at course level
    course_agg = transactions.groupby("CourseID").agg(
        EnrollmentCount=("TransactionID", "count"),
        TotalRevenue=("Amount", "sum"),
        AvgRevenue=("Amount", "mean"),
    ).reset_index()
    course_agg["RevenuePerEnrollment"] = course_agg["TotalRevenue"] / course_agg["EnrollmentCount"]

    # Merge courses + teacher via transactions (TeacherID in transactions)
    teacher_per_course = transactions.groupby("CourseID")["TeacherID"].first().reset_index()
    courses = courses.merge(teacher_per_course, on="CourseID", how="left")
    courses = courses.merge(teachers[["TeacherID","Expertise","YearsOfExperience","TeacherRating"]], on="TeacherID", how="left")
    df = courses.merge(course_agg, on="CourseID", how="left")
    df.fillna({"EnrollmentCount": 0, "TotalRevenue": 0, "AvgRevenue": 0, "RevenuePerEnrollment": 0}, inplace=True)

    return df, transactions

def engineer_features(df):
    # Price bands
    df["PriceBand"] = pd.cut(df["CoursePrice"], bins=[-1, 0, 150, 350, 1000],
                              labels=["Free", "Low", "Medium", "High"])
    # Duration buckets
    df["DurationBucket"] = pd.cut(df["CourseDuration"], bins=[0, 15, 30, 50, 200],
                                   labels=["Short", "Medium", "Long", "Extended"])
    # Rating tiers
    df["RatingTier"] = pd.cut(df["CourseRating"], bins=[0, 2, 3, 4, 5],
                               labels=["Low", "Medium", "High", "Top"])
    # Experience buckets
    df["ExpBucket"] = pd.cut(df["YearsOfExperience"], bins=[0, 3, 7, 12, 50],
                              labels=["Junior", "Mid", "Senior", "Expert"])
    # Expertise-category match score
    df["ExpertiseMatch"] = (df["Expertise"] == df["CourseCategory"]).astype(int)

    # Encode
    cat_cols = ["CourseCategory", "CourseType", "CourseLevel", "PriceBand", "DurationBucket",
                "RatingTier", "ExpBucket"]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    return df, df_enc

def get_feature_matrix(df_enc, target="EnrollmentCount"):
    drop_cols = ["CourseID", "CourseName", "TeacherID", "Expertise",
                 "EnrollmentCount", "TotalRevenue", "AvgRevenue", "RevenuePerEnrollment"]
    drop_cols = [c for c in drop_cols if c in df_enc.columns]
    X = df_enc.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df_enc[target]
    return X, y
