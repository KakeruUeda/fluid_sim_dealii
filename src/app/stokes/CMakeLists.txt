set(main_target "StokesFlow")

# --- sources ---------
set(SOURCES
    main.cc
)

add_executable(${main_target} ${SOURCES})
deal_ii_setup_target(${main_target})

install(TARGETS ${main_target} DESTINATION ${CMAKE_INSTALL_PREFIX})