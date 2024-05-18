
def yes_no_question(question, default=None):
    """Ask yes/no question and record user input.
    
    Parameters
    ----------
    question : str
        Question to ask user
    default : bool, optional
        Default answer if user input is left blank
    
    Returns
    -------
    answer : bool
        User's answer to yes/no question
    """

    if default is None:
        yn = ' (y/n) '
    elif default is True:
        yn = ' ([y]/n) '
    elif default is False:
        yn = ' (y/[n]) '
    else:
        raise TypeError('Argument `default` must be bool or None'
            f" but is of type '{type(default)}'")

    while True:
        answer = input(question + yn).strip().lower()
        if not answer:
            if default is not None:
                return default
        elif answer in ('y','yes'):
            return True
        elif answer in ('n','no'):
            return False
