#!/usr/bin/env python3
"""
⏱️ SPRINT TIMER 4H - CRITEO INTERVIEW PREP
Timer Pomodoro optimisé pour session 4 heures
"""

import time
import datetime
import os
import sys
from typing import List, Dict


class SprintTimer4H:
    """Timer pour sprint de préparation 4 heures"""

    def __init__(self):
        self.phases = [
            {"name": "🚀 Setup & Mental Prep", "duration": 10, "break": 0},
            {"name": "💪 DSA - Hash/Array", "duration": 25, "break": 0},
            {"name": "💪 DSA - Graph/BFS", "duration": 30, "break": 5},
            {"name": "📊 SQL Analytics", "duration": 35, "break": 5},
            {"name": "🤖 CTR & ML Core", "duration": 30, "break": 5},
            {"name": "🚀 DeepKNN & Retrieval", "duration": 35, "break": 5},
            {"name": "💰 Bidding & Auctions", "duration": 20, "break": 0},
            {"name": "🎯 Fairness & System", "duration": 20, "break": 0},
            {"name": "🏁 Pitch & Integration", "duration": 20, "break": 0},
            {"name": "✅ Closing Review", "duration": 5, "break": 0}
        ]

        self.current_phase = 0
        self.start_time = None
        self.phase_start = None
        self.completed_tasks = []

    def clear_screen(self):
        """Clear console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_progress_bar(self, current, total, width=50):
        """Display a progress bar"""
        progress = current / total if total > 0 else 0
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = progress * 100
        return f"[{bar}] {percentage:.1f}%"

    def format_time(self, seconds):
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def display_header(self):
        """Display header with overall progress"""
        self.clear_screen()
        print("="*80)
        print("⏱️  CRITEO INTERVIEW PREP - 4H SPRINT TIMER")
        print("="*80)

        if self.start_time:
            elapsed = time.time() - self.start_time
            total_duration = sum(p['duration'] + p['break'] for p in self.phases)
            overall_progress = min(elapsed / (total_duration * 60), 1.0)

            print(f"\n🕐 Session Time: {self.format_time(elapsed)}")
            print(f"📊 Overall Progress: {self.display_progress_bar(elapsed, total_duration * 60)}")

        print("\n" + "-"*80)

    def show_phase_tasks(self, phase_name):
        """Show tasks for current phase"""
        tasks = {
            "DSA - Hash/Array": [
                "✓ Two Sum All Pairs problem",
                "✓ Sliding Window Maximum",
                "✓ Subarray Sum Equals K"
            ],
            "DSA - Graph/BFS": [
                "✓ User Network Clustering",
                "✓ Campaign Dependencies (Topological Sort)",
                "✓ Click Fraud Detection"
            ],
            "SQL Analytics": [
                "✓ Rolling Window CTR",
                "✓ Multi-touch Attribution",
                "✓ Cohort Analysis"
            ],
            "CTR & ML Core": [
                "✓ Hash Trick Implementation",
                "✓ Feature Engineering",
                "✓ Metrics Calculation (LogLoss, AUC)"
            ],
            "DeepKNN & Retrieval": [
                "✓ Two-Tower Architecture",
                "✓ Faiss Indexing",
                "✓ Latency Optimization"
            ],
            "Bidding & Auctions": [
                "✓ Bid Shading Calculation",
                "✓ First-Price Strategy",
                "✓ Budget Pacing"
            ],
            "Fairness & System": [
                "✓ Demographic Parity",
                "✓ System Design Template",
                "✓ Scale Calculations"
            ],
            "Pitch & Integration": [
                "✓ 60-Second Pitch",
                "✓ Top 5 Questions",
                "✓ Speed Review"
            ]
        }

        # Extract simplified name
        simple_name = phase_name.split(" ", 1)[1] if " " in phase_name else phase_name

        if simple_name in tasks:
            print("\n📋 TASKS FOR THIS PHASE:")
            for task in tasks[simple_name]:
                print(f"   {task}")

    def run_phase(self, phase: Dict):
        """Run a single phase with timer"""
        phase_name = phase['name']
        duration = phase['duration']

        self.phase_start = time.time()
        end_time = self.phase_start + (duration * 60)

        print(f"\n{'='*80}")
        print(f"🎯 PHASE {self.current_phase + 1}/10: {phase_name}")
        print(f"⏱️  Duration: {duration} minutes")
        print("="*80)

        self.show_phase_tasks(phase_name)

        print(f"\n⏰ Timer starts NOW! Press Ctrl+C to skip phase.")
        print("-"*80)

        try:
            while time.time() < end_time:
                remaining = end_time - time.time()
                elapsed = time.time() - self.phase_start

                # Update display every second
                self.display_header()
                print(f"\n🎯 Current Phase: {phase_name}")
                print(f"⏱️  Time Remaining: {self.format_time(remaining)}")
                print(f"📊 Phase Progress: {self.display_progress_bar(elapsed, duration * 60)}")

                # Show warnings
                if remaining < 60:
                    print("\n⚠️  LAST MINUTE! Wrap up!")
                elif remaining < 300:
                    print(f"\n⏰ {int(remaining/60)} minutes remaining")

                # Quick tips based on phase
                self.show_phase_tips(phase_name, remaining)

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n⏭️  Phase skipped!")

        # Phase complete
        self.completed_tasks.append(phase_name)
        print(f"\n✅ Phase '{phase_name}' completed!")

        # Break time
        if phase['break'] > 0:
            self.take_break(phase['break'])

    def show_phase_tips(self, phase_name, remaining):
        """Show contextual tips based on phase and time"""
        tips = {
            "DSA - Hash/Array": "💡 Remember: Two-pointer pattern, Hash for complement lookup",
            "DSA - Graph/BFS": "💡 Remember: Use deque for BFS, visited set for cycles",
            "SQL Analytics": "💡 Key: PARTITION BY, ORDER BY, ROWS BETWEEN",
            "CTR & ML Core": "💡 Targets: LogLoss < 0.44, AUC > 0.80",
            "DeepKNN & Retrieval": "💡 Key: Two-tower + Faiss, <50ms p99",
            "Bidding & Auctions": "💡 Remember: Shading factor 0.75-0.85",
            "Pitch & Integration": "💡 Keep it under 60 seconds!"
        }

        simple_name = phase_name.split(" ", 1)[1] if " " in phase_name else phase_name
        if simple_name in tips and remaining > 10:
            print(f"\n{tips[simple_name]}")

    def take_break(self, break_duration):
        """Break between phases"""
        print(f"\n{'='*80}")
        print(f"☕ BREAK TIME - {break_duration} minutes")
        print("Stand up, stretch, hydrate!")
        print("="*80)

        end_time = time.time() + (break_duration * 60)

        try:
            while time.time() < end_time:
                remaining = end_time - time.time()
                print(f"\r⏱️  Break remaining: {self.format_time(remaining)}   ", end="")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏭️  Break skipped!")

        print("\n🔔 Break over! Back to work!")

    def show_final_summary(self):
        """Display final summary"""
        self.clear_screen()
        total_time = time.time() - self.start_time

        print("="*80)
        print("🎉 SPRINT COMPLETED!")
        print("="*80)

        print(f"\n⏱️  Total Time: {self.format_time(total_time)}")
        print(f"✅ Phases Completed: {len(self.completed_tasks)}/10")

        print("\n📋 COMPLETED PHASES:")
        for task in self.completed_tasks:
            print(f"   ✓ {task}")

        print("\n🎯 FINAL CHECKLIST:")
        checklist = [
            "DSA patterns mastered",
            "SQL window functions clear",
            "CTR pipeline understood",
            "DeepKNN architecture ready",
            "Bid shading calculated",
            "Fairness metrics known",
            "60-second pitch practiced",
            "Questions prepared"
        ]

        for item in checklist:
            response = input(f"   {item}? (y/n): ").strip().lower()
            if response == 'y':
                print(f"   ✅ {item}")
            else:
                print(f"   ❌ {item} - Review this!")

        print("\n💪 YOU'RE READY FOR THE INTERVIEW!")
        print("Remember: Confidence > Perfection")
        print("\n🚀 GO CRUSH IT!")

    def run(self):
        """Run the complete 4-hour sprint"""
        print("="*80)
        print("🚀 CRITEO INTERVIEW PREP - 4 HOUR SPRINT")
        print("="*80)

        print("\n📝 BEFORE WE START:")
        print("   ✓ Phone on airplane mode?")
        print("   ✓ Water and snacks ready?")
        print("   ✓ IDE and files open?")
        print("   ✓ Comfortable environment?")

        ready = input("\n✅ Ready to start? (y/n): ").strip().lower()
        if ready != 'y':
            print("Take your time to prepare. Run again when ready!")
            return

        print("\n🎯 YOUR 3 GOALS FOR TODAY:")
        for i in range(1, 4):
            goal = input(f"   Goal {i}: ")

        print("\n⚡ LET'S DO THIS!")
        print("Sprint starts in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("🚀 GO!")

        self.start_time = time.time()

        # Run all phases
        for i, phase in enumerate(self.phases):
            self.current_phase = i
            self.run_phase(phase)

        # Final summary
        self.show_final_summary()


def quick_mode():
    """Quick 1-hour focused session"""
    print("="*80)
    print("⚡ QUICK MODE - 1 HOUR ESSENTIALS")
    print("="*80)

    phases = [
        {"name": "DSA Core Problems", "duration": 20},
        {"name": "CTR & DeepKNN", "duration": 20},
        {"name": "Bidding & Pitch", "duration": 15},
        {"name": "Quick Review", "duration": 5}
    ]

    print("\n📋 1-HOUR PLAN:")
    for i, phase in enumerate(phases, 1):
        print(f"   {i}. {phase['name']} ({phase['duration']} min)")

    print("\n⏰ Starting in 3 seconds...")
    time.sleep(3)

    start = time.time()
    for phase in phases:
        print(f"\n{'='*60}")
        print(f"🎯 {phase['name']} - {phase['duration']} minutes")
        print("="*60)
        print("Timer running... Press Ctrl+C to skip")

        try:
            time.sleep(phase['duration'] * 60)  # Simplified for quick mode
        except KeyboardInterrupt:
            print("Skipped!")

    elapsed = (time.time() - start) / 60
    print(f"\n✅ Quick session complete! ({elapsed:.1f} minutes)")


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_mode()
    else:
        timer = SprintTimer4H()
        timer.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Timer stopped by user")
        print("Remember to review what you've learned!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please report this issue!")