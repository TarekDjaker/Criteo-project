#!/usr/bin/env python3
"""
📊 CRITEO INTERVIEW PREP - PROGRESS DASHBOARD
Interactive dashboard to track your preparation progress
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
import time


class CriteoProgressDashboard:
    def __init__(self):
        self.progress_file = "progress.json"
        self.data = self.load_progress()
        self.topics = {
            "DSA": {"total": 20, "completed": 0, "priority": 5},
            "SQL": {"total": 10, "completed": 0, "priority": 4},
            "CTR_Modeling": {"total": 5, "completed": 0, "priority": 5},
            "DeepKNN": {"total": 3, "completed": 0, "priority": 5},
            "Bidding": {"total": 4, "completed": 0, "priority": 4},
            "System_Design": {"total": 3, "completed": 0, "priority": 4},
            "Behavioral": {"total": 5, "completed": 0, "priority": 3},
            "Criteo_Specific": {"total": 10, "completed": 0, "priority": 5}
        }

    def load_progress(self) -> Dict:
        """Load existing progress or create new"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "start_date": datetime.now().isoformat(),
            "total_hours": 0,
            "problems_solved": [],
            "weak_areas": [],
            "strong_areas": [],
            "notes": []
        }

    def save_progress(self):
        """Save progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def display_dashboard(self):
        """Display the main dashboard"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("=" * 80)
        print("📊 CRITEO INTERVIEW PREP - PROGRESS DASHBOARD")
        print("=" * 80)
        print(f"\n📅 Started: {self.data['start_date'][:10]}")
        print(f"⏱️  Total Study Hours: {self.data['total_hours']:.1f}h")
        print(f"✅ Problems Solved: {len(self.data['problems_solved'])}")

        # Progress bars for each topic
        print("\n📈 TOPIC PROGRESS:")
        print("-" * 80)

        overall_progress = 0
        overall_total = 0

        for topic, info in self.topics.items():
            completed = info["completed"]
            total = info["total"]
            priority = info["priority"]
            progress = (completed / total) * 100 if total > 0 else 0

            overall_progress += completed
            overall_total += total

            # Create progress bar
            bar_length = 30
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Priority stars
            stars = "⭐" * priority

            # Color coding
            if progress >= 80:
                status = "✅"
            elif progress >= 50:
                status = "🟡"
            else:
                status = "🔴"

            print(f"{status} {topic:15} [{bar}] {progress:5.1f}% ({completed}/{total}) {stars}")

        # Overall progress
        overall_percent = (overall_progress / overall_total) * 100 if overall_total > 0 else 0
        print("\n" + "=" * 80)
        print(f"🎯 OVERALL PROGRESS: {overall_percent:.1f}% ({overall_progress}/{overall_total})")

        # Recommendations
        self.show_recommendations()

        # Study streak
        self.show_study_streak()

    def show_recommendations(self):
        """Show study recommendations based on progress"""
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 80)

        # Find weakest areas
        weak_areas = []
        for topic, info in self.topics.items():
            progress = (info["completed"] / info["total"]) * 100 if info["total"] > 0 else 0
            priority = info["priority"]

            # Weight by priority
            weighted_gap = (100 - progress) * priority / 5
            if weighted_gap > 30:
                weak_areas.append((topic, weighted_gap, priority))

        weak_areas.sort(key=lambda x: x[1], reverse=True)

        if weak_areas:
            print("⚠️  Focus Areas (by priority):")
            for i, (topic, gap, priority) in enumerate(weak_areas[:3], 1):
                topic_clean = topic.replace("_", " ")
                print(f"   {i}. {topic_clean} (Priority: {'⭐' * priority})")
        else:
            print("✅ Great progress across all areas!")

        # Time-based recommendations
        hour = datetime.now().hour
        if 6 <= hour < 12:
            print("\n🌅 Morning Session Recommended:")
            print("   - Complex DSA problems (fresh mind)")
            print("   - System design thinking")
        elif 12 <= hour < 17:
            print("\n☀️  Afternoon Session Recommended:")
            print("   - SQL practice")
            print("   - CTR modeling implementation")
        else:
            print("\n🌙 Evening Session Recommended:")
            print("   - Review concepts")
            print("   - Behavioral prep")

    def show_study_streak(self):
        """Display study streak information"""
        print("\n🔥 STUDY STREAK:")
        print("-" * 80)

        # Calculate streak (simplified)
        if "last_study_date" in self.data:
            last_study = datetime.fromisoformat(self.data["last_study_date"])
            today = datetime.now()
            days_diff = (today - last_study).days

            if days_diff == 0:
                print("✅ Studied today! Keep it up!")
            elif days_diff == 1:
                print("⚠️  Last studied yesterday. Don't break the streak!")
            else:
                print(f"❌ Last studied {days_diff} days ago. Time to get back!")
        else:
            print("🚀 Start your streak today!")

    def log_study_session(self):
        """Log a study session"""
        print("\n📝 LOG STUDY SESSION")
        print("-" * 40)

        # Topic selection
        print("\nSelect topic studied:")
        topics_list = list(self.topics.keys())
        for i, topic in enumerate(topics_list, 1):
            print(f"{i}. {topic.replace('_', ' ')}")

        choice = input("\nEnter number (1-8): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= 8:
            selected_topic = topics_list[int(choice) - 1]

            # Number of items completed
            items = input(f"How many {selected_topic.replace('_', ' ')} items completed? ").strip()
            if items.isdigit():
                self.topics[selected_topic]["completed"] += int(items)

                # Hours studied
                hours = input("Hours studied: ").strip()
                if hours.replace('.', '').isdigit():
                    self.data["total_hours"] += float(hours)

                # Notes
                notes = input("Notes (optional): ").strip()
                if notes:
                    self.data["notes"].append({
                        "date": datetime.now().isoformat(),
                        "topic": selected_topic,
                        "note": notes
                    })

                self.data["last_study_date"] = datetime.now().isoformat()
                self.save_progress()
                print("\n✅ Session logged successfully!")
            else:
                print("❌ Invalid input")
        else:
            print("❌ Invalid choice")

    def quick_test(self):
        """Quick knowledge test"""
        print("\n🧠 QUICK KNOWLEDGE TEST")
        print("-" * 40)

        questions = [
            {
                "q": "What is Criteo's dataset size for CTR?",
                "a": "45M samples, 13 numerical + 26 categorical features"
            },
            {
                "q": "What is the target latency for DeepKNN?",
                "a": "< 50ms p99"
            },
            {
                "q": "When did the industry move to first-price auctions?",
                "a": "2019"
            },
            {
                "q": "What is the Demographic Parity threshold in FairJob?",
                "a": "< 0.7%"
            },
            {
                "q": "What are the two main components of DeepKNN?",
                "a": "Two-tower encoders + Vector DB (Faiss)"
            }
        ]

        import random
        question = random.choice(questions)

        print(f"\n❓ {question['q']}")
        input("\nPress Enter to see answer...")
        print(f"\n✅ Answer: {question['a']}")

        correct = input("\nDid you get it right? (y/n): ").strip().lower()
        if correct == 'y':
            print("🎉 Great job!")
        else:
            print("📚 Review this topic!")

    def export_report(self):
        """Export progress report"""
        print("\n📄 GENERATING PROGRESS REPORT...")

        report = []
        report.append("=" * 60)
        report.append("CRITEO INTERVIEW PREP - PROGRESS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("=" * 60)

        report.append(f"\n📊 SUMMARY:")
        report.append(f"Total Hours: {self.data['total_hours']:.1f}h")
        report.append(f"Problems Solved: {len(self.data['problems_solved'])}")

        report.append(f"\n📈 PROGRESS BY TOPIC:")
        for topic, info in self.topics.items():
            progress = (info["completed"] / info["total"]) * 100 if info["total"] > 0 else 0
            report.append(f"  {topic}: {progress:.1f}% ({info['completed']}/{info['total']})")

        report.append(f"\n📝 RECENT NOTES:")
        for note in self.data.get("notes", [])[-5:]:
            date = note["date"][:10]
            report.append(f"  [{date}] {note['topic']}: {note['note']}")

        report.append(f"\n✅ READY FOR INTERVIEW: {'YES' if self.calculate_readiness() > 80 else 'NOT YET'}")

        # Save report
        report_file = f"progress_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))

        print(f"✅ Report saved to {report_file}")

    def calculate_readiness(self) -> float:
        """Calculate overall readiness percentage"""
        total_progress = 0
        total_weight = 0

        for topic, info in self.topics.items():
            progress = (info["completed"] / info["total"]) * 100 if info["total"] > 0 else 0
            weight = info["priority"]

            total_progress += progress * weight
            total_weight += weight

        return total_progress / total_weight if total_weight > 0 else 0

    def run(self):
        """Main dashboard loop"""
        while True:
            self.display_dashboard()

            print("\n" + "=" * 80)
            print("📋 ACTIONS:")
            print("1. Log Study Session")
            print("2. Quick Knowledge Test")
            print("3. Export Progress Report")
            print("4. Reset Topic Progress")
            print("5. Exit")

            choice = input("\nSelect action (1-5): ").strip()

            if choice == "1":
                self.log_study_session()
                input("\nPress Enter to continue...")
            elif choice == "2":
                self.quick_test()
                input("\nPress Enter to continue...")
            elif choice == "3":
                self.export_report()
                input("\nPress Enter to continue...")
            elif choice == "4":
                confirm = input("\n⚠️  Reset all progress? (yes/no): ").strip().lower()
                if confirm == "yes":
                    for topic in self.topics:
                        self.topics[topic]["completed"] = 0
                    self.data["total_hours"] = 0
                    self.data["problems_solved"] = []
                    self.save_progress()
                    print("✅ Progress reset!")
                input("\nPress Enter to continue...")
            elif choice == "5":
                print("\n👋 Good luck with your preparation!")
                break
            else:
                print("❌ Invalid choice")
                time.sleep(1)


def main():
    """Launch the dashboard"""
    dashboard = CriteoProgressDashboard()

    # ASCII art welcome
    print("""
    ╔═══════════════════════════════════════╗
    ║   CRITEO INTERVIEW PREP DASHBOARD    ║
    ║          Track Your Progress          ║
    ║         Ace That Interview!           ║
    ╚═══════════════════════════════════════╝
    """)

    time.sleep(2)
    dashboard.run()


if __name__ == "__main__":
    main()