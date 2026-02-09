def run_challenge_four():
    from challenge_4_object_tracking.main import main
    main()

if __name__ == "__main__":
    print("Computer Vision - Guide 1")
    print("1. Matrices to Images")
    print("3. Colors")
    option = input("Select challenge: ")

    if option == "1":
        run_challenge_1()
    elif option == "4":
        run_challenge_four
