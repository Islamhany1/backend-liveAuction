package com.liveauction.backend_liveAuction.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.ArrayList;

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtUtil jwtUtil;

    public JwtAuthenticationFilter(JwtUtil jwtUtil) {
        this.jwtUtil = jwtUtil;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {

        // 1. Look for the "Authorization" header
        String authHeader = request.getHeader("Authorization");

        // 2. Check if it's missing or isn't a Bearer token
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            // No badge? Move along. Spring Security will reject them later if the room is locked.
            filterChain.doFilter(request, response);
            return;
        }

        // 3. Extract the raw token string (Skip the 7 characters of "Bearer ")
        String token = authHeader.substring(7);

        try {
            // 4. Verify the math and get the username
            String username = jwtUtil.extractUsername(token);

            // 5. If valid, and the whiteboard is currently blank
            if (username != null && SecurityContextHolder.getContext().getAuthentication() == null) {

                // Create the official Spring Security ID Badge
                UsernamePasswordAuthenticationToken authToken = new UsernamePasswordAuthenticationToken(
                        username,
                        null,
                        new ArrayList<>() // This is where "Roles" (like ADMIN/USER) would go
                );

                // 6. Write their name on the whiteboard!
                SecurityContextHolder.getContext().setAuthentication(authToken);
            }
        } catch (Exception e) {
            // If the token is expired or forged, Jwts.parser() throws an error.
            // We catch it and leave the whiteboard blank. Access Denied.
            System.out.println("Token rejected: " + e.getMessage());
        }

        // 7. Crucial: Tell the request to continue down the chain to the Controller
        filterChain.doFilter(request, response);
    }
}