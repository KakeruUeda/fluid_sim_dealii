set(main_target "NavierStokesFlow")

# --- sources ---------
set(SOURCES
    main.cc
)

add_executable(${main_target} ${SOURCES})
deal_ii_setup_target(${main_target})

target_include_directories(${main_target} 
PRIVATE
  ${PROJECT_SOURCE_DIR}/src/lib/utils
  ${PROJECT_SOURCE_DIR}/src/lib/forward
)

install(TARGETS ${main_target} DESTINATION ${CMAKE_INSTALL_PREFIX})