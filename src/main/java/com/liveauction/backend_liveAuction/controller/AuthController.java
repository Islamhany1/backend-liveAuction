package com.liveauction.backend_liveAuction.controller;

import com.liveauction.backend_liveAuction.entity.User;
import com.liveauction.backend_liveAuction.repository.UserRepository;
import com.liveauction.backend_liveAuction.security.JwtUtil;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.Optional;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtil jwtUtil;

    public AuthController(UserRepository userRepository, PasswordEncoder passwordEncoder, JwtUtil jwtUtil) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtUtil = jwtUtil;
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@RequestBody RegisterRequest request) {

        if (userRepository.findByUsername(request.username()).isPresent()) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(new ApiResponse("Username is already taken."));
        }

        User newUser = new User();
        newUser.setUsername(request.username());
        newUser.setEmail(request.email());
        newUser.setPassword(passwordEncoder.encode(request.password()));

        userRepository.save(newUser);

        return ResponseEntity.ok(new ApiResponse("User registered successfully!"));
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request) {

        Optional<User> optionalUser = userRepository.findByUsername(request.username());

        if (optionalUser.isEmpty()) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(new ApiResponse("Invalid username or password."));
        }

        User user = optionalUser.get();

        if (!passwordEncoder.matches(request.password(), user.getPassword())) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(new ApiResponse("Invalid username or password."));
        }

        String token = jwtUtil.generateToken(user.getUsername());

        return ResponseEntity.ok()
                .header("Authorization", "Bearer " + token)
                .body(new ApiResponse("Login successful!"));
    }
}

// DTOs
record RegisterRequest(String username, String email, String password) {}
record LoginRequest(String username, String password) {}
record ApiResponse(String message) {}