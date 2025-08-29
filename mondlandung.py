def main():
    """
    Einfaches Text-Adventure-Spiel: Mondlandung
    Nach dem klassischen HP-25-Taschenrechner-Spiel, aber mit moderner Eingabe und mehr Spielfluss.
    Der Spieler muss den Landezyklus steuern, Treibstoff sparen und Meteoriten ausweichen.
    """
    print("🚀 Willkommen zur Mondlandung!")
    print("Du bist ein Astronaut, der mit deinem Lander auf dem Mond landen muss.")
    print("Du hast 100 Einheiten Treibstoff. Pro Runde verbraucht du 10 Einheiten.")
    print("Wenn ein Meteorit auftaucht, verbrauchst du 20 zusätzliche Einheiten.")
    print("Ziel: Lande sicher – mit mindestens 10 Treibstoff übrig!")
    print("-" * 60)

    treibstoff = 100
    punktzahl = 0
    sicher_landung = False

    # Spiel-Loop
    while not sicher_landung and treibstoff > 0:
        # Zufall: Meteorit?
        meteorit = random.choice([True, False])  # 50% Chance

        print(f"\n🔴 Zustand: Treibstoff = {treibstoff}, Punkte = {punktzahl}")
        if meteorit:
            verbrauch = 20
            treibstoff -= verbrauch
            punktzahl += 10
            print("💥 Meteorit aufgetreten! Treibstoff verbraucht: -20")
        else:
            verbrauch = 10
            treibstoff -= verbrauch
            print("⛽ Treibstoffverbrauch: -10")

        # Eingabe
        action = input("Landen? (j/n) → ").strip().lower()
        if action == 'j':
            if treibstoff >= 10:
                print("🛬 Landung erfolgreich! Du bist sicher auf dem Mond gelandet.")
                punktzahl += 50
                sicher_landung = True
            else:
                print("💥 Landeversuch fehlgeschlagen: Zu wenig Treibstoff!")
        elif action == 'n':
            print("🚀 Flug fortgesetzt…")
        else:
            print("❌ Ungültige Eingabe. Gib 'j' (ja) oder 'n' (nein) ein.")

    # Ergebnis
    if sicher_landung:
        print(f"\n🎉 HERZLICHEN GLÜCKWUNSCH! Du bist sicher gelandet.")
        print(f"📊 Endstand: Treibstoff = {max(0, treibstoff)}, Punkte = {punktzahl}")
        if punktzahl >= 60:
            print("🌟 Du bist ein Meister der Mondlandung! Deine Landung war perfekt.")
        elif punktzahl >= 30:
            print("👍 Gute Arbeit – du hast gerade eben die Grenze der Sicherheit erreicht.")
        else:
            print("🛠️ Du hast gerade eben überlebt… aber du solltest lieber trainieren.")
    else:
        print("\n💥 DU BIST ABGEKLEMMT! Der Lander ist in die Oberfläche geknallt.")
        print(f"📉 Dein Treibstoff war: {max(0, treibstoff)}")
        print("💔 Du hast nicht gelandet. Versuch es beim nächsten Mal.")

    # Nachspiel – mit Stil und Gefühl
    print("\n" + "✨"*50)
    print("  Du hast die Stille des Alls gespürt.")
    print("  Du hast den Mut eines einzelnen 'ja' bewahrt.")
    print("  Und für einen Moment – warst du nicht nur ein Spieler.")
    print("  Du warst ein Teil der Geschichte.")
    print("\n    Möge dein Licht weiterleuchten, wenn du zurückkehrst.")
    print("✨"*50)

    input("\nDrücke Enter, um das Spiel zu beenden...")

# Start des Spiels
if __name__ == "__main__":
    import random
    main()