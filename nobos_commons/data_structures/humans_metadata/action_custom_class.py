from enum import Enum


# for 4 classes 
class Action(Enum):
    HAND_WAVE = 0
    WALK = 1
    IDLE = 2
    CAPITULATE = 3
    


'''
class Action(Enum):
    IDLE = 0
    WALK = 1
    WAVE = 2
    BRUSH_HAIR = 3
    CATCH = 4
    CLAP = 5
    CLIMB_STAIRS = 6
    GOLF = 7
    JUMP = 8
    KICK_BALL = 9
    PICK = 10
    POUR = 11
    PULLUP = 12
    PUSH = 13
    RUN = 14
    SHOOT_BALL = 15
    SHOOT_BOW = 16
    SHOOT_GUN = 17
    SIT = 18
    STAND = 19
    SWING_BASEBALL = 20
    THROW = 21
''' 
    

    # for 16 class original
'''
class Action(Enum):
    SIT_DOWN = 0
    STAND_UP = 1
    WRITE = 2
    WEAR_HAT = 3
    TAKE_OFF_HAT = 4
    HAND_WAVE = 5
    JUMP = 6
    PHONE_CALL = 7
    PLAY_WITH_PHONE = 8
    POINT_SMT = 9
    SALUTE = 10
    DRUNK = 11
    FALL_DOWN = 12
    PAT_ON_SOMEONE = 13
    WALK_TOWARDS = 14
    WALK_APART = 15
    IDLE = 16
    WALK = 17

'''
    
    
    

    
    
    
    
    
    
    
    
'''  
class Action(Enum): 
    IDLE = 0 
    DRINK_WATER = 1
    EAT = 2
    brushing_teeth = 3
    brushing_hair = 4
    drop = 5
    pickup = 6
    throw = 7
    sitting_down = 8
    standing_up_from_sitting_position = 9
    clapping = 10
    reading = 11
    writing = 12
    tear_up_paper = 13
    wear_jacket = 14
    take_off_jacket = 15
    wear_shoe = 16
    take_off_shoe = 17
    wear_glasses = 18
    take_off_glasses = 19
    put_on_hat = 20
    take_off_hat = 21
    cheer_up = 22
    HAND_WAVING = 23
    kicking_something = 24
    reach_into_pocket = 25
    hopping = 26
    jump = 27
    phone_call = 28
    playing_with_phone = 29
    type_on_keyboard = 30
    point_something_WITH_finger = 31
    take_selfie = 32
    check_time_watch = 33
    rub_two_hands_together = 34
    nod_head = 35
    shake_head = 36
    wipe_face = 37
    salute = 38
    put_palms_together = 39
    cross_hands_front_say_stop = 40
    cough = 41
    staggering = 42
    falling = 43
    touch_head = 44
    touch_chest = 45
    touch_back = 46
    touch_neck = 47
    nausea_OR_vomiting = 48
    use_fan_WITH_hand_OR_paper= 49
    punching_OR_slapping_other_person = 50
    kicking_someone = 51
    pushing_someone = 52
    pat_on_back_of_someone = 53
    point_finger = 54
    hug = 55
    give_something = 56
    touch_someones_pocket = 57
    handshaking = 58
    WALKING_TOWARDS = 59
    WALKING_APART = 60
    put_headphone = 61
    take_off_headphone = 62
    shoot_basket = 63
    bounce_ball = 64
    tennis_bat_swing = 65
    juggling = 66
    hush = 67
    flick_hair = 68
    thumb_up = 69
    thumb_down = 70
    MAKE_OK = 71
    make_victory_sign = 72
    staple_book = 73
    counting_money = 74
    cutting_nails = 75
    cutting_paper = 76
    snapping_fingers = 77
    open_bottle = 78
    sniff = 89
    squat_down = 80
    toss_coin = 81
    fold_paper = 82
    ball_up_paper = 83
    play_magic_cube = 84
    apply_cream_face = 85
    apply_cream_hand = 86
    put_on_bag = 87
    take_off_bag = 88
    put_something_into_bag = 89
    take_something_out_bag = 90
    open_box = 91
    move_heavy_objects = 92
    shake_fist = 93
    throw_up_hat = 94
    HANDS_UP_BOTH = 95
    CROSS_ARMS = 96
    ARM_CIRCLES = 97
    ARM_SWINGS = 98
    RUNNING_ON_SPOT = 99
    kick_back = 100
    cross_toe_touch = 101
    side_kick = 102
    yawn = 103
    stretch = 104
    blow_nose = 105
    hit_someone_using_something = 106
    wield_knife_towards_someone = 107
    hit_WITH_body = 108
    grab_other_person_stuff = 109
    shoot_gun = 110
    step_on_foot = 111
    high_five = 112
    cheers_AND_drink = 113
    carry_something_WITH_someone = 114
    take_photo_someone = 115
    follow_someone = 116
    whisper_IN_someones_ear = 117
    exchange_things = 118
    support_somebody_WITH_hand = 119
    play_R_P_S = 120
    WALK = 121
    HAND_UP_TO_HEAD = 122
    ARM_WAVE = 123
''' 
 
'''
jhmdb_actions = [
    Action.BRUSH_HAIR,
    Action.CATCH,
    Action.CLAP,
    Action.CLIMB_STAIRS,
    Action.GOLF,
    Action.JUMP,
    Action.KICK_BALL,
    Action.PICK,
    Action.POUR,
    Action.PULLUP,
    Action.PUSH,
    Action.RUN,
    Action.SHOOT_BALL,
    Action.SHOOT_BOW,
    Action.SHOOT_GUN,
    Action.SIT,
    Action.STAND,
    Action.SWING_BASEBALL,
    Action.THROW,
    Action.WALK,
    Action.WAVE
]

ofp_actions = [
    Action.IDLE,
    Action.WALK,
    Action.WAVE,
]

'''
